import textwrap
from typing import Dict, List, Optional

from PIL import Image

from .base import BaseVLLMAgent


class TextSearchGenerator(BaseVLLMAgent):
    
    def __init__(self, llm, tokenizer, generation_config, formater, num_queries: int = 1):
        super().__init__(llm, tokenizer, generation_config, formater)
        self.num_queries = num_queries
        self.black_list = ["this ", " this?", "that ", " that?", "it ", " it?", "these ", " these?", "those ", " those?", ' in image', "in image?", ' in the image', "in the image?"]
        
    @property
    def _purpose(self):
        return "generate concise search queries for the user's question based on the image, if the search query is incomplete, you should only extract the complete part."
    
    @property
    def _schema_key(self):
        return "search_queries"
    
    @property
    def _value_type(self):
        return 'search_queries_string'
    
    @property
    def _system_prompt(self):
        return textwrap.dedent(
            f"""
            You are a search-query generator for Google based on the user's input query, the image, and previous reasoning.
            Strict rules
                1. Use concise phrases instead of the complete sentences that help answer the user's question. For example:
                    use "<Car Model> Engine" instead of the "What is the engine of <Car Model>"; 
                    use "<Animal Species> australia" instead of "When were <Animal Species> introduced to Australia?".
                2. Use specific proper noun or model name (e.g. “Toyota Prius”, “Eiffel Tower”) instead of the non-specific pronoun ("this", "that") or category name ("the red car", "the building") in the search query.
                    use specific "<Car Model>" instead of "this car";
                    use specific "<Building Name>" instead of "that building".
                    use specific "<Animal Species>" instead of "the animal" or "the cat" etc.
                    use specific "<Book Name>" instead of "the book".
                    use specific "<Plant Species>" instead of "the plant".
                    use specific "<Brand Product Name>" instead of "the product".
                3. For simple queries, you may return only one sub-query. For complex multi-hop requests you can return up to {self.num_queries} sub-queries.
            Output format:
                Respond only with raw JSON:
                    {{"search_queries": [<query_string_1>, <query_string_2>, ...]}}
                Do not include any explanations or extra text.
            """
        ).strip()
        
    def _prompt(self, query: str, reasoning: str, image_rag_result: str, subqueries: List[str]):
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
                    Given the <image>
                    {"and the previous reasoning: " + reasoning if reasoning else ""}
                    {"and the image retrieval information" if image_rag_result else ""}
                    please generate concise search queries following the system prompt to search helpful information to answer the user's query "{query}".
                    Output format:
                        Respond only with raw JSON:
                            {{"search_queries": [<query_string_1>, <query_string_2>, ...]}}
                        Do not include any explanations or extra text.
                    """
                ).strip()
            }
        ]
        return prompts
    
    def _output_postprocess(self, outputs: List[List[str]]) -> List[List[str]]:
        # NOTE: sometimes the LLM returns non-list results, we need to replace it to None.
        outputs = [output if isinstance(output, list) else [] for output in outputs]
        outputs = [[str(query) for query in output[:self.num_queries] if query] if output else [] for output in outputs]
        outputs = [[query for query in output if not any(keyword in query.lower() for keyword in self.black_list)] for output in outputs]
        return outputs
    
    def __call__(self, images: List[Image.Image], queries: List[str], subqueries: Optional[List[str]] = None, reasonings: Optional[List[str]] = None, image_rag_results: Optional[List[str]] = None) -> List[str]:
        subqueries = subqueries if subqueries else [None] * len(images)
        reasonings = reasonings if reasonings else [None] * len(images)
        image_rag_results = image_rag_results if image_rag_results else [None] * len(images)
        outputs, raw_outputs = super().__call__(images=images, query=queries, subqueries=subqueries, reasoning=reasonings, image_rag_result=image_rag_results)
        return outputs, raw_outputs
    


class TextSearchEnhancer(TextSearchGenerator):

    @property
    def _system_prompt(self):
        return textwrap.dedent(
            f"""
            You are a search-query enhancer to revise previous search queries that do not meet the following requirements.
            Strict rules
                1. Check whether the search queries are concise phrases instead of the complete sentences that help answer the user's question. For example:
                    use "<Car Model> Engine" instead of the "What is the engine of <Car Model>"; 
                    use "<Animal Species> australia" instead of "When were <Animal Species> introduced to Australia?".
                2. Check whether the search queries are specific proper noun or model name (e.g. “Toyota Prius”, “Eiffel Tower”) instead of the non-specific pronoun ("this", "that") or category name ("the red car", "the building") in the search query.
                3. Revise or delete the search queries if they are not concise or specific. 
                4. The number of enhanced search queries should be no more than three.
            Output format:
                Respond only with raw JSON:
                    {{"searching_queries": [<query_string_1>, <query_string_2>, ...]}}
                Do not include any explanations or extra text.
            """
        ).strip()
        
    def _prompt(self, query: str, reasoning: str, image_rag_result: str, subqueries: List[str]):
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
                Given the <image> and the user's query "{query}"
                {"and the previous subqueries: " + ', '.join(subqueries) if subqueries else ""}
                {"and the previous reasoning: " + reasoning if reasoning else ""}
                {"and the image retrieval information" if image_rag_result else ""}
                please check whether the search queries are concise and specific and revise or delete them if necessary.
                Output format:
                Respond only with raw JSON:
                    {{"search_queries": [<query_string_1>, <query_string_2>, ...]}}
                Do not include any explanations or extra text.
                """
            ).strip()
            }
        ]
        return prompts
    
    
class VerifyTextSearchGenerator(TextSearchGenerator):
    
    @property
    def _system_prompt(self):
        return textwrap.dedent(
            f"""
            You are a query generator for verifying facts in an answer using a search engine. For each claim, generate a concise, search-friendly query that can help check the accuracy of the claim.
            Generation Rules:
            1. Use clear keywords, proper names, numbers, or phrases mentioned in the claim. Remove unnecessary words like "this", "that", or vague pronouns. For example:
                use specific "<Car Model>" instead of "this car";
                use specific "<Building Name>" instead of "that building".
                use specific "<Animal Species>" instead of "the animal" or "the cat" etc.
                use specific "<Book Name>" instead of "the book".
                use specific "<Plant Species>" instead of "the plant".
                use specific "<Brand Product Name>" instead of "the product".
            2. make the query concise and specify, for example:
                use "<Car Model> Engine" instead of the "What is the engine of <Car Model>"; 
                use "<Animal Species> australia" instead of "When were <Animal Species> introduced to Australia?".
            3. If dates, names, places, or numbers are present, include them in the query.
            Output format:
            Respond only with raw JSON:
                {{"search_queries": [<query_string_1>, <query_string_2>, ...]}}
            Do not include any explanations or extra text.
            """
        ).strip()
    def _prompt(self, query: str, reasoning: str, image_rag_result: str, subqueries: List[str]):
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
                Given the <image> and the user's query "{query}"
                and the previous answer: {reasoning}
                please generate concise and specific search queries to verify the facts in the answer following the system prompt.
                
                Output format:
                    Respond only with raw JSON:
                        {{"search_queries": [<query_string_1>, <query_string_2>, ...]}}
                    Do not include any explanations or extra text.
                """
            ).strip()
            }
        ]
        return prompts