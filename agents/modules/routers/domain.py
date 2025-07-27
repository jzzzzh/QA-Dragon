import textwrap
from typing import List

from PIL import Image

from ..base import BaseVLLMAgent


class DomainRouter(BaseVLLMAgent):
    
    def __init__(self, llm, tokenizer, generation_config, formater):
        super().__init__(llm, tokenizer, generation_config, formater)
    
    @property 
    def _purpose(self):
        return """decide the question domain in ["living beings", "food", "landmarks", "text or chart reasoning", "shopping product info", "vehicle", "other"]."""

    @property
    def _schema_key(self):
        return "domain"
    @property
    def _value_type(self):
        return 'domain_class_string'
    
    @property
    def _system_prompt(self):
        return textwrap.dedent("""
                You are a visual assistant that identify the question domain based on the query and image.
                The question domain should be one of the following: "living beings", "food", "landmarks", "text or chart reasoning", "shopping product info", "vehicle", and "other".
                Where:
                    living beings: Questions about animals or plants—species ID, traits, behaviour, habitat, growth or care needs, taxonomy, and biological comparisons.
                    Examples:
                        1. is this cat more likely to be a male or female?
                        2. what's the difference between the fruit of this plant and that of Ophiopogon planiscapus?
                        3. in 2020, what percent of the world's total production of this did the world's largest producer make?
                    food: Questions about dishes, ingredients, nutrition, cooking methods, or the cultural/industrial origin of food items.
                    Examples:
                        1. what is the origin of this food item?
                        2. how are these different from Hachiya persimmons?
                        3. when was the company that makes this product founded?
                    landmarks: Questions about notable buildings, monuments, or geographic sites—location, designers, construction timeline, historic alterations, or related artists.
                    Examples:
                        1. how long did this take to build?
                        2. when was the artist who made this born?
                        3. what are the two times the building underwent external remodeling?
                    text or chart reasoning: Questions that require reading and interpreting text, tables, charts, diagrams or formulas, often with numeric extraction or scientific reasoning.
                    Examples:
                        1. which age and sex had the least amount of injury-related unintentional falls reported?
                        2. how many hydrogen atoms are in this?
                        3. what is the angle of this?
                    shopping product info: Questions about consumer goods or published media—price, specifications, packaging, availability, editions, or author/publisher details.
                    Examples:
                        1. how many calories does this have per serving?
                        2. which version of this book is cheaper, the hardcover or the e-book?
                        3. who are the authors of the latest edition of this book?

                    vehicle: Questions about cars, trains, aircraft, boats, etc.—make, model, capacity, performance, range calculations, or historical comparisons.
                    Examples:
                        1. how many passengers can the red car seat?
                        2. can the 2023 model with a 1.3 L AWD engine of this vehicle get from Washington DC to Baltimore on 5 gallons of gas?
                        3. was this car around before the Jaguar XK-E?
                    other: Questions that don't clearly fit into the above categories.
                If you are not sure about the question domain, you should return "other".
                Output your predicted domain in JSON format:
                {"domain": "<domain_class_string>"}
                """
            ).strip()
    
    @property
    def domains(self):
        return ["living beings", "food", "landmarks", "text or chart reasoning", "shopping product info", "vehicle", "other"]
    
    def _prompt(self, query: str):
        user_prompt = textwrap.dedent(f"""
                Given the <image> and query text: {query}
                Output your predicted domain in JSON format like:
                {{"domain": <domain_class_string>}}
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
        return [str(output) if output in self.domains else "other" for output in outputs]
    
    def __call__(self, images: List[Image.Image], queries: List[str]) -> List[str]:
        return super().__call__(images=images, query=queries)