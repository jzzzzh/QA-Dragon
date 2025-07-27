import textwrap
from typing import List
from agents.modules.base import BaseVLLMAgent
from PIL import Image
from typing import Optional


class ComplexityRouter(BaseVLLMAgent):    
    def __init__(self, llm, tokenizer, generation_config, formater, use_image: bool = False):
        super().__init__(llm, tokenizer, generation_config, formater)
        self.use_image = use_image

    @property 
    def _purpose(self):
        return "determine the complexity of the given questions."
    
    @property
    def _schema_key(self):
        return "complexity"
    
    @property
    def _value_type(self):
        return '<easy/hard>'
    
    @property
    def _system_prompt(self):
        return textwrap.dedent(f"""
                You are an expert reasoning complexity classifier.
                Given a question related to an image, determine the reasoning complexity required to answer it.

                Decision Rules:
                    1.	Easy Reasoning (≤2-Hop):
                The question can be answered via direct image recognition or by retrieving no more than 2 simple facts, attributes, or relationships.
                These require 1 or 2 reasoning steps only.
                Examples:
                    - "When was this logo introduced?" — requires recognizing the logo and retrieving the introduction date (2 hops).
                    - "What park was this restaurant’s first location in?" — requires recognizing the restaurant and retrieving its first location (2 hops).
                    2.	Hard Reasoning (>2-Hop):
                The question requires more than 2 facts or reasoning steps, involving complex or multi-fact connections.
                Examples:
                    - "Can the 2023 model with a 1.3L AWD engine of this vehicle get from Washington, DC to Baltimore on 5 gallons of gas?" — requires fuel efficiency, distance, and fuel amount (3 hops).
                    - "How old is the company that makes this car?" — requires recognizing the car, identifying the company, and calculating its age (3 hops).
                Output the complexity in JSON format:
                {{"complexity": <easy/hard>}}
                No other text or explanation is allowed.
                """
            ).strip()
    
    def _prompt(self, queries: str):
        user_prompt = textwrap.dedent(f"""
                Given the query text: {queries}
                What is the complexity of the given reasoning text?
                """
            ).strip()
        
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
                "content": user_prompt
            },
        ]

        if not self.use_image:
            prompts.pop(1)
        return prompts
    
    def _output_postprocess(self, outputs: List[str]) -> List[str]:
        complexities = list()
        for output in outputs:
            if isinstance(output, str):
                complexities.append(output.strip().lower())
            else:
                complexities.append("unknown")
        return complexities

    def __call__(self, images: Optional[List[Image.Image]] = None, queries: List[str] = None) -> List[bool]:
        if not self.use_image:
            images = None
        return super().__call__(images=images, queries=queries)
