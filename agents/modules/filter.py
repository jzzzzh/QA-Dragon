import textwrap
from typing import Any, Dict, List, Tuple

import nltk
from PIL import Image

from .base import BaseVLLMAgent


class ImageSearchFilter(BaseVLLMAgent):
    
    def __init__(self, llm, tokenizer, generation_config, formater):
        super().__init__(llm, tokenizer, generation_config, formater)
        
    @property
    def _purpose(self):
        return "identify the object in the image from the image search results"
    
    @property
    def _schema_key(self):
        return "object"
    
    @property
    def _value_type(self):
        return 'object_name_string'
    
    @property
    def _system_prompt(self):
        return textwrap.dedent("""
            You are an image-understanding assistant.
            Look at <image> and compare what you see with the candidate object names provided in this message.
            Return the single candidate that is visually present. If no candidate is present, return null.
            Decision rules
                1. Rely only on visual evidence (shape, logo, context, etc.).
                2. Ignore any object that is not listed among the candidates.
                3. The candidate objects may be all incorrect. If the match is uncertain, choose null rather than guessing.
                4. Output must be one line of raw JSON with the exact key "object_name".
            Required output format
                {"object_name":"<candidate_name_or_null>"}
                No other text should be output.
                """
            ).strip()
        
    def _prompt(self, object_name_list: List[str], segment_category: str):
        user_prompt = textwrap.dedent(f"""
                Given the <image> and a list of candidate object names: {object_name_list}
                which {segment_category} object in the given list is the object in the image?
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
        return outputs
    
    @staticmethod
    def _get_entity_name(item: Dict[str, Any]) -> str:
        try:
            return item['entities'][0]['entity_name']
        except:
            return ''
    
    def __call__(self, images: List[Image.Image], search_results: List[List[Dict[str, Any]]], segment_categories: List[str]) -> Tuple[List[str], List[List[Dict[str, Any]]], List[str]]:
        # This function filters search results for each image and segment category, selecting the most visually matching object name (or null if none match).
        # It returns three lists: the filtered object names, the corresponding filtered search result items, and the raw filtered object names or error messages.
        assert len(search_results) == len(images) == len(segment_categories), (
            f"{len(search_results) = }, {len(images) = }, {len(segment_categories) = }"
        )
        total_filtered_object_names = [None] * len(images)
        total_raw_filtered_object_names = ['empty result'] * len(images)
        total_filtered_img_search = [[] for _ in images]

        # Indices where search_results are non-empty
        have_results_index = [i for i, search_result in enumerate(search_results) if search_result]
        object_name_lists = [[self._get_entity_name(item) for item in search_results[i]] for i in have_results_index]
        object_image_lists = [images[i] for i in have_results_index]
        filtered_segment_categories = [segment_categories[i] for i in have_results_index]

        # Call parent filter to get filtered object names
        filtered_object_names, raw_filtered_object_names = super().__call__(images=object_image_lists,object_name_list=object_name_lists,segment_category=filtered_segment_categories)

        # Map filtered results back to original indices
        for i, filtered_object_name, raw_filtered_object_name in zip(
            have_results_index, filtered_object_names, raw_filtered_object_names
        ):
            if filtered_object_name:
                total_filtered_object_names[i] = filtered_object_name
                total_raw_filtered_object_names[i] = raw_filtered_object_name
                seen = set()
                for item in search_results[i]:
                    if self._get_entity_name(item) == filtered_object_name:
                        item_str = str(item)
                        if item_str not in seen:
                            total_filtered_img_search[i].append(item)
                            seen.add(item_str)
            else:
                total_raw_filtered_object_names[i] = f'No filtered object name {raw_filtered_object_name}'

        return total_filtered_object_names, total_filtered_img_search, total_raw_filtered_object_names
    
class QueryFilter(object):
    
    def __init__(self):
        self.black_list = set(
            ['size', 'described', 'area', 'years', 'church', 'habitat', 'invented', 'orchid', 'conservation', 'ingredients', 'last', 'bridge', 'population', 'versus', 'differ', 'endemic', 'sculpture', 'period', 'serving', 'total', 'sales', 'design', 'plastic', 'expensive', 'house', 'since', 'temple', 'formally', 'birds', 'latest', 'floors', 'pasta', 'commonly', 'percentage', 'modern', 'hours', 'across', 'models', '’', 'botanist', 'shell', 'typical', 'version', 'price', 'flowers', 'color', 'difference', 'world', 'current', 'us', 'fish', 'largest', 'created', 'suv', 'owns', 'book', 'breed', 'tall', 'style', 'available', 'president', 'generations', 'go', 'native', 'common', 'born', 'long', 'museum', 'large', 'work', 'two', 'range', 'number', 'built', 'soil', 'grams',]
        )
    def __call__(self, queries: List[str]) -> Tuple[List[str], List[int], List[bool]]:
        tokenized_queries = [set(nltk.word_tokenize(q)) for q in queries]
        filter_flag = [q & self.black_list for q in tokenized_queries]
        filter_flag = [bool(flag) or any(any(char.isdigit() for char in word) for word in nltk.word_tokenize(q)) for flag, q in zip(filter_flag, queries)]
        filtered_index = [i for i, flag in enumerate(filter_flag) if flag]
        remaining_queries = [q for q, flag in zip(queries, filter_flag) if not flag]
        return remaining_queries, filtered_index, filter_flag
