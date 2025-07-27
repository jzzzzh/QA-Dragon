import textwrap
from typing import List, Tuple

import vllm
from PIL import Image

from ..base import BaseVLLMAgent


class ImageTextSearchToolRouter(BaseVLLMAgent):
    
    def __init__(self, llm, tokenizer, generation_config, formater):
        super().__init__(llm, tokenizer, generation_config, formater)
    
    @property 
    def _purpose(self):
        return "decide whether to call the `image_search` and/or `text_search` tools."
    
    @property
    def _schema_key(self):
        return ["need_image_search", "need_text_search"]
    
    @property
    def _value_type(self):
        return ['<true/false>', '<true/false>']
    
    @property
    def _system_prompt(self):
        return textwrap.dedent(f"""
            You are an action-selector. Given three inputs - (1) the user's query, (2) any prior reasoning text, and (3) the image - decide in a single turn which retrieval tool or tools must run before the answer is generated. You may choose image_search, text_search, both, neither.
            Tools:
            {self._tool_description}
            Decision logic
                1. Do you or the reasoning text already know the object's specific identity (proper noun or model name)?
                    - Yes: set `need_image_search` to false.
                    - No: set `need_image_search` to true.
                2. Does the query need additional information that are not visible in the image (specifications, history, statistics, price, etc.)?
                    - Yes: set `need_text_search` to true.
                    - No: set `need_text_search` to false.
                3. If it is about address some scientific calculation queries like math, physics, etc. or language translation, set both flags to false.
                4. If the object is a "book", a "logo-bearing packaged goods", or "plant", set `need_image_search` to false.
                Both flags may be true; when so, run image_search first, then text_search.
            
            Produce exactly one sentence to conduct the decision logic. This is the only non-JSON text allowed.
            Immediately after the sentence, output a single valid JSON object:
            Decision logic: <concise explanation>
            Tool calling decision: {{"need_image_search": <true/false>, "need_text_search": <true/false>}}
            Do not output anything else.
            
            Here are some examples:
            Example 1:
                User:
                    Image: a car on the road.
                    Query: "What is the price of the car?"
                    Reasoning: "I identified the object in the image, it is a <Car Model> car. But the price is not visible in the image."
                Assistant:
                    Decision logic: "I know the object in the image, it is a <Car Model> car. So I don't need to search the image again. But the price is not visible in the image. So I need to search the text to get the price."
                    Tool calling decision: {{"need_image_search": false, "need_text_search": true}}
            Example 2:
                User:
                    Image: a car on the road.
                    Query: "What is the price of the car?"
                    Reasoning: "I cannot identify the object in the image."
                Assistant:
                    Decision logic: "I cannot identify the object in the image. So I need to search the image to get the object's identity. Assume I got the object's identity, the price is not visible in the image. So I need to search the text to get the price."
                    Tool calling decision: {{"need_image_search": true, "need_text_search": true}}
            Example 3:
                User:
                    Image: a car on the road.
                    Query: "What is the nickname of the car?"
                    Reasoning: "I identified the object in the image, it is a <Car Model> car. The nickname of the car is <Car Nickname>."
                Assistant:
                    Decision logic: "I know the object in the image, it is a <Car Model> car. So I don't need to search the image again. Also I know the nickname of the <Car Model> car. So I don't need to search the text to get the nickname."
                    Tool calling decision: {{"need_image_search": false, "need_text_search": false}}
            """
            ).strip()
    
    @property
    def _tool_description(self):
        return """
                Tool #1 image_search
                [Description] Retrieve visually similar images via embeddings to identify an object whose specific name is still unknown.
                [Use when] The object in the picture is not known or known only by a generic label (e.g. “car”, “jacket”, “statue”) instead of a specific name/model/species in previous reasoning.
                [Input] The input image.
                [Output] Text snapshots (top-k) from wikipedia or amazon that show visually similar objects.

                Tool #2 text_search
                [Description] Issue a refined natural-language web query to fetch textual facts about an object whose specific name is already known.
                [Use when] The query requires the additional information not available in the image.
                [Input] A text query constructed from the user's question + the known identity.
                [Output] Text snippets (top-k) from relevant websites.
            """ 
    
    def _prompt(self, query: str, reasoning: str):
        prompts = [
            {
                "role": "system",
                "content": self._system_prompt
            },
            {
                "role": "user",
                "content": [{"type": "image"}]
            },
            {
                "role": "user",
                "content": textwrap.dedent(
                    f"""
                    Query: {query}
                    Previous Reasoning: {reasoning}
                    What tools should I run? `image_search`, `text_search`, both, or neither? Please conduct the decision logic and output your choice following the system prompt.
                    """
                ).strip()
            }
        ]
        return prompts
    
    def _output_postprocess(self, image_search_tool_flags: List[bool], text_search_tool_flags: List[bool]):
        processed = list()
        for image_search_tool_flag, text_search_tool_flag in zip(image_search_tool_flags, text_search_tool_flags):
            # process image_search_tool_flag
            if isinstance(image_search_tool_flag, bool):    
                pass
            else:
                try:
                    if isinstance(image_search_tool_flag, str):
                        val = image_search_tool_flag.strip().lower()
                        if val == "true":
                            image_search_tool_flag = True
                        elif val == "false" or val == "none":
                            image_search_tool_flag = False
                        else:
                            raise ValueError(f"Cannot convert string to bool: {image_search_tool_flag}")
                    else:
                        image_search_tool_flag = bool(image_search_tool_flag)
                except Exception as e:
                    print(f"[RuleAnswerableRouter] Exception in _output_postprocess: {e}. Setting to True.")
                    image_search_tool_flag = True
            # process text_search_tool_flag
            if isinstance(text_search_tool_flag, bool):    
                pass
            else:
                try:
                    if isinstance(text_search_tool_flag, str):
                        val = text_search_tool_flag.strip().lower()
                        if val == "true":
                            text_search_tool_flag = True
                        elif val == "false" or val == "none":
                            text_search_tool_flag = False
                        else:
                            raise ValueError(f"Cannot convert string to bool: {text_search_tool_flag}")
                    else:
                        text_search_tool_flag = bool(text_search_tool_flag)
                except Exception as e:
                    print(f"[RuleAnswerableRouter] Exception in _output_postprocess: {e}. Setting to True.")
                    text_search_tool_flag = True
            processed.append((image_search_tool_flag, text_search_tool_flag))
        return processed
    
    def __call__(self, images: List[Image.Image], queries: List[str], reasonings: List[str]) -> List[Tuple[bool, bool]]:
        self._check_batching(images=images, queries=queries, reasonings=reasonings)

        prompts = [
            {
                "prompt": self.tokenizer.apply_chat_template(
                self._prompt(query, reasoning),
                add_generation_prompt=True,
                tokenize=False),
            } 
            for query, reasoning in zip(queries, reasonings)
        ]
        
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
        
        text_outputs = [output.outputs[0].text for output in outputs]
        image_search_tool_flags = self.formater(
            batch_text=text_outputs, 
            purpose=self._purpose, 
            schema_key=self._schema_key[0], 
            value_type=self._value_type[0]
        )
        text_search_tool_flags = self.formater(
            batch_text=text_outputs, 
            purpose=self._purpose, 
            schema_key=self._schema_key[1], 
            value_type=self._value_type[1]
        )
        return self._output_postprocess(image_search_tool_flags, text_search_tool_flags), text_outputs
    