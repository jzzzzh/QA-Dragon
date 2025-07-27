import abc
import math
from typing import Any, Dict, List, Optional

import numpy as np
import vllm
from PIL import Image

from ..utils.formater import Formater


class BaseVLLMAgent(abc.ABC):
    
    def __init__(self, llm, tokenizer, generation_config, formater: Formater):
        self.llm = llm
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.formater = formater
    
    @property
    def _purpose(self):
        raise NotImplementedError("Subclasses must implement this property")
    
    @property
    def _value_type(self):
        raise NotImplementedError("Subclasses must implement this property")
    
    @property
    def _schema_key(self):
        raise NotImplementedError("Subclasses must implement this property")
    
    @property
    def _system_prompt(self):
        raise NotImplementedError("Subclasses must implement this property")
    
    @abc.abstractmethod
    def _prompt(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abc.abstractmethod
    def _output_postprocess(self, output: str):
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(self, images: Optional[List[Image.Image]] = None, use_ori_outputs=False, **prompt_kwargs):
        self._check_batching(images, **prompt_kwargs)

        prompts = [
            {
                "prompt": self.tokenizer.apply_chat_template(
                self._prompt(**prompt_kwarg),
                add_generation_prompt=True,
                tokenize=False),
            } 
            for prompt_kwarg in self._batch_prompt_kwargs(prompt_kwargs)
            ]
        # TODO: add to return list.
        self.prompts = prompts
        
        # add image to the prompts if it is provided.
        if images:
            for prompt, image in zip(prompts, images):
                prompt["multi_modal_data"] = {"image": image}

        outputs = self.llm.generate(
            prompts,
            sampling_params=vllm.SamplingParams(
                temperature=self.generation_config['temperature'],
                top_p=self.generation_config['top_p'],
                max_tokens=self.generation_config['max_tokens'],
                logprobs=self.generation_config.get('logprobs', None),
                skip_special_tokens=True,
            ),
        )

        if use_ori_outputs:
            outputs = [output.outputs[0] for output in outputs]
            raw_text_outputs = [output.text for output in outputs]
            return self._output_postprocess(outputs), raw_text_outputs
        
        raw_text_outputs = [output.outputs[0].text for output in outputs]
        if self.formater:
            text_outputs = self.formater(
                batch_text=raw_text_outputs, 
                purpose=self._purpose, 
                schema_key=self._schema_key, 
                value_type=self._value_type
            )
        else:
            text_outputs = raw_text_outputs
        return self._output_postprocess(text_outputs), raw_text_outputs
    
    @staticmethod
    def _check_batching(images: Optional[List[Image.Image]] = None, **prompt_kwargs):
        if images:
            if not isinstance(images, list):
                raise ValueError(f"images must be a list, but got {type(images)}")
            n = len(images)
            for _, v in prompt_kwargs.items():
                if not isinstance(v, list):  
                    raise ValueError(f"All prompt_kwargs must be lists when image is provided, but got {type(v)} for one of the keys")
                elif len(v) != n:
                    raise ValueError(f"All prompt_kwargs must be lists of the same length {n} when image is provided, but got {len(v)} for one of the keys")
        else:
            lens = [len(v) for v in prompt_kwargs.values() if isinstance(v, list)]
            if lens:
                n = lens[0]
                for k, v in prompt_kwargs.items():
                    if not isinstance(v, list) or len(v) != n:
                        raise ValueError(f"All prompt_kwargs must be lists of the same length {n} when batching without image, but got {len(v)} for {k}")
    
    @staticmethod
    def _batch_prompt_kwargs(prompt_kwargs: Dict[str, Any]):
        """
        Batch prompt kwargs:
        Given prompt_kwargs = {'arg1': [a1, a2], 'arg2': [b1, b2]}
        Output: [{'arg1': a1, 'arg2': b1}, {'arg1': a2, 'arg2': b2}]
        """
        if not prompt_kwargs:
            return []
        batch_size = len(next(iter(prompt_kwargs.values())))
        for i in range(batch_size):
            yield {k: v[i] for k, v in prompt_kwargs.items()}
    
    @staticmethod   
    def _calculate_certainty(output: Dict[str, Any]) -> Dict[str, float]:
        logprobs = []
        for logprob, token_id in zip(output.logprobs, output.token_ids):
            logprobs.append(logprob[token_id].logprob)
        
        norm_prob = math.exp(np.mean(logprobs))
        min_prob = min([math.exp(lp) for lp in logprobs])

        certainty_score = {
            "norm_prob": norm_prob,
            "min_prob": min_prob
        }
        return certainty_score

   
class BaseInfoAwareVLLMAgent(BaseVLLMAgent):
    """
    Base class for VLLM agents that are aware of additional information.
    This class extends BaseVLLMAgent to include an additional `key` parameter.
    """
    
    @abc.abstractproperty
    def info_list(self) -> List[str]:
        raise NotImplementedError("Subclasses must implement this property")
    
    @abc.abstractmethod
    def _system_prompt(self, info: str) -> str:
        raise NotImplementedError("Subclasses must implement this function")
    
    @abc.abstractmethod
    def _prompt(self, info: str, **kwargs):
        raise NotImplementedError("Subclasses must implement this function")