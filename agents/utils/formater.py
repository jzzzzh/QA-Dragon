import re
from typing import List

import textwrap
import json_repair
import vllm


class Formater(object):
    
    
    def __init__(self, llm, tokenizer, generation_config, reformat_with_llm=True):
        self.llm = llm
        self.tokenizer = tokenizer
        self.generation_config = generation_config or dict()
        self.reformat_with_llm = reformat_with_llm

    def __call__(self, batch_text: List[str], purpose: str, schema_key: str, value_type: str):
        json_objects = [self.parse_json(t, schema_key) for t in batch_text]
        mask = [obj is None for obj in json_objects]

        if any(mask) and self.reformat_with_llm:
            schema_str = '{' + f'"{schema_key}": "<{value_type}>"' + '}'
            reformat_prompts = []
            for idx, t in enumerate(batch_text):
                if mask[idx]:
                    prompt = self.tokenizer.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that reformats text into valid JSON according to a given schema."
                            },
                            {
                                "role": "user",
                                "content": textwrap.dedent(f"""
                                    The following is an attempt to {purpose}, but it may not be valid JSON:
                                    {t}
                                    Please reformat it into valid JSON in the following format:
                                    {schema_str}
                                """).strip()
                            }
                        ],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    reformat_prompts.append({"prompt": prompt})

            if reformat_prompts:
                outputs = self.llm.generate(
                    reformat_prompts,
                    sampling_params=vllm.SamplingParams(
                        temperature=self.generation_config.get("temperature", 0.0),
                        top_p=self.generation_config.get("top_p", 1.0),
                        max_tokens=self.generation_config.get("max_tokens", 128),
                        skip_special_tokens=True,
                    ),
                )
                reformatted_idx = 0
                for idx, failed in enumerate(mask):
                    if failed:
                        reformatted_text = outputs[reformatted_idx].outputs[0].text
                        json_objects[idx] = self.parse_json(reformatted_text, schema_key)
                        reformatted_idx += 1

        return json_objects

    def parse_json(self, text, schema_key: str):
        try:
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                obj_json = json_repair.loads(match.group(0))
                if schema_key in obj_json and (isinstance(obj_json[schema_key], bool) or obj_json[schema_key]):
                    return obj_json[schema_key]
        except Exception:
            pass
        return None