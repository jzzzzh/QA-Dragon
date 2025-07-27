import textwrap
from typing import Dict, List

from PIL import Image

from .base import BaseVLLMAgent


class ObjectDetectionAgent(BaseVLLMAgent):

    def __init__(self, llm, tokenizer, generation_config, formater, object_num: int = 3):
        super().__init__(llm, tokenizer, generation_config, formater)
        self.object_num = object_num

    @property
    def _purpose(self):
        return "recognize key objects in an image relevant to a question"

    @property
    def _schema_key(self):
        return "object_list"

    @property
    def _value_type(self):
        return "object_name_string"

    @property
    def _system_prompt(self):
        return "You are an expert AI system for object detection and identification. Your task is to recognize and list high-level object categories shown in an image that are relevant to a given question. Return only structured results in JSON format."

    def _prompt(self, query: str):
        user_prompt = textwrap.dedent(
            f"""
                Identify and list up to {self.object_num} major distinct objects in the image that are visually present and relevant to the question: "{query}". 
        
                Only include tangible, visible items (e.g., "car", "brand", "clothing", "book", "device", "food", "building"). 
                Do not include abstract concepts (e.g., "emotion", "relationship") or actions (e.g., "running", "shopping").
                Use general categories, not specific names: "BMW" to "car", "ZARA" to "clothing brand", "iPhone 13" to "smartphone", "Coca-Cola" to "drink"
                Each object name should be short with no more than 3 words.
                If unsure of the exact identity, use the closest general category (e.g., "electronic device", "building", "plant").
                Return the result in the following JSON format strictly:
                {{"object_list": ["<object_name_string>", "<object_name_string>", ...]}}
                """
        ).strip()
        prompts = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": [{"type": "image"}]},
            {"role": "user", "content": user_prompt},
        ]
        return prompts

    def _output_postprocess(self, outputs: List[List[str]]) -> List[str]:
        return [output if isinstance(output, list) else ['mainObject'] for output in outputs]

    def __call__(self, images: List[Image.Image], queries: List[str]) -> List[str]:
        return super().__call__(images=images, query=queries)
    


class ObjectSelectionAgent(BaseVLLMAgent):
    
    def __init__(self, llm, tokenizer, generation_config, formater):
        super().__init__(llm, tokenizer, generation_config, formater)

    @property
    def _purpose(self):
        return "select the one most relevant object from a list of objects"

    @property
    def _schema_key(self):
        return "object"

    @property
    def _value_type(self):
        return "object_name_string"

    @property
    def _system_prompt(self):
        return "You are an AI assistant to select one object from a list of objects, which is most relevant to the object queried by the question. Only return structured results in JSON format."

    def _prompt(self, query: str, object_list: List[str]):
        user_prompt = textwrap.dedent(
            f"""
                Given the image and list of objects detected in the image: {object_list}
                and the question: "{query}"
                select the one object in the object list that the question is about.
                If the query includes position-related words, give priority to objects at or near the position.
                Give a short sentence about the reason to choose the object and return the final selected object in this format:
                {{"object": "<object_name_string>"}}
                """
        ).strip()
        prompts = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": [{"type": "image"}]},
            {"role": "user", "content": user_prompt},
        ]
        return prompts

    def _output_postprocess(self, outputs: List[str]) -> List[str]:
        return [str(output) if output else 'mainObject' for output in outputs]

    def __call__(self, images: List[Image.Image], queries: List[str], object_list: List[List[str]]) -> List[str]:
        return super().__call__(images=images, query=queries, object_list=object_list)
    
    
class PresetObjectSelectionAgent(BaseVLLMAgent):
    
    def __init__(self, llm, tokenizer, generation_config, formater):
        super().__init__(llm, tokenizer, generation_config, formater)

    @property
    def _purpose(self):
        return "select the one most relevant object"

    @property
    def _schema_key(self):
        return "object"

    @property
    def _value_type(self):
        return "object_name_string"

    @property
    def _preset_object_list(self):
        return ["vehicle", "animals", "building", "merchandise", "brand logo", "statue or sculpture", "art", "book", "screen", "fruit", "plant", "food"]

    @property
    def _system_prompt(self):
        return textwrap.dedent(f"""
            You are an AI assistant to select one object in the image from the list: {self._preset_object_list} that is most relevant to the input query.
            Selection Criteria:
                1. The object must be visible in the image and clearly related to the query.
                    * If a hand appears, assume the target object is the item held or immediately nearby.
                    * If the query explicitly mentions a word from the preset list, choose that object.
                2. If several identical instances of the chosen object are present in the image, add an adjective that uniquely identifies the most relevant one (e.g., “red car”, “white dog”, “green fruit”, “the tallest building”).
            Output Format:
                Only output the most relevant one object in the following JSON format:
                    {{"object": "<object_name_string>"}}
                No other words are allowed.
        """).strip()

    def _prompt(self, query: str):
        user_prompt = textwrap.dedent(
            f"""
                Given the input query "{query}" for the <image> 
                select the one object in the preset list following the Selection Criteria and Output Format. No other words are allowed.
                """
        ).strip()
        prompts = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": [{"type": "image"}]},
            {"role": "user", "content": user_prompt},
        ]
        return prompts

    def _output_postprocess(self, outputs: List[str]) -> List[str]:
        return [str(output) if output else 'mainObject' for output in outputs]

    def __call__(self, images: List[Image.Image], queries: List[str]) -> List[str]:
        return super().__call__(images=images, query=queries)