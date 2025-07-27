import textwrap
from typing import List

from PIL import Image

from .base import BaseVLLMAgent


class QueryRephraseAgent(BaseVLLMAgent):
    
    def __init__(self, llm, tokenizer, generation_config, formater):
        super().__init__(llm, tokenizer, generation_config, formater)
        
    @property
    def _purpose(self):
        return "rephrase the user's question to a more specific and clear question"
    
    @property
    def _schema_key(self):
        return "query"
    
    @property
    def _value_type(self):
        return 'query_string'
    
    @property
    def _system_prompt(self):
        return textwrap.dedent("""
                You are a visual assistant that rephrases vague or ambiguous user questions into clearer and more specific versions.
                Rules:

                In your rephrased query, always use the specific name of the item mentioned or shown in the image (e.g., “the Toyota Camry”, “the giraffe”, “the apple tree”), instead of generic terms like “the car” or “the animal”. For example, rephrase “when was the car made” as “when was the Toyota Camry made”.
                Preserve the original question's structure and intent, including multi-hop reasoning, but make the wording more precise and unambiguous.
                Output your rephrased query in JSON format:
                {"query": "<query_string>"}
                """
            ).strip()
        
    def _prompt(self, query: str):
        user_prompt = textwrap.dedent(f"""
                Given the <image> and query text: {query}
                Output your rephrased query in JSON format like:
                {{"query": <query_string>}}
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
        return prompts
    
    def _output_postprocess(self, outputs: List[str]) -> List[str]:
        return [str(output) if output else "I don't know" for output in outputs]
    
    def __call__(self, images: List[Image.Image], queries: List[str]) -> List[str]:
        return super().__call__(images=images, query=queries)
    