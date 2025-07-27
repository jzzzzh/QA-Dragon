import textwrap
from typing import List, Dict, Any, Optional

from PIL import Image

from .base import BaseVLLMAgent, BaseInfoAwareVLLMAgent


class BasicReasoningAgent(BaseVLLMAgent):
    
    def __init__(self, llm, tokenizer, generation_config, formater):
        super().__init__(llm, tokenizer, generation_config, formater)
        
    @property
    def _purpose(self):
        return "conduct reasoning to answer the user's question based on the image"
    
    @property
    def _schema_key(self):
        return "reasoning"
    
    @property
    def _value_type(self):
        return 'your_reasoning_string'
    
    @property
    def _system_prompt(self):
        return textwrap.dedent("""
                You are a visual assistant that use step-by-step reasoning based on the image to address the user's question. Keep your reasoning concise as less as possible with no more than 3 steps and only one sentence for each step. 
                Output your reason in JSON format like:
                {"reasoning": <your_reasoning_string>}
                """
            ).strip()
        
    def _prompt(self, query: str):
        user_prompt = textwrap.dedent(f"""
                Given the <image> and query text: {query}
                Output your reasoning based on the image and the text query in JSON format like:
                {{"reasoning": <your_reasoning_string>}}
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


class DomainAwareReasoningAgent(BaseInfoAwareVLLMAgent):
    # TODO: make the domain examples' query consistent with the user prompt.
    def __init__(self, llm, tokenizer, generation_config, formater):
        super().__init__(llm, tokenizer, generation_config, formater)
        
    @property
    def _purpose(self):
        return "conduct reasoning to answer the user's question based on the image"
    
    @property
    def _schema_key(self):
        return "reasoning"
    
    @property
    def _value_type(self):
        return 'your_reasoning_string'
    
    @property
    def info_list(self) -> List[str]:
        return ["living beings", "food", "landmarks", "text or chart reasoning", "shopping product info", "vehicle", "other"]
    
    def _domain_system_prompt(self, domain: str) -> Dict[str, Dict[str, str]]:
        domain_system_prompt = {
            "living beings": {
                "reasoning_guide": textwrap.dedent("""
                Domain Reasoning Guidelines:
                    1. You can specify the plant or animal based on their visible characteristics.
                    2. The specific name should be precise and specific to the species.
                """),
            },
            "food": {
                "reasoning_guide": textwrap.dedent("""
                Domain Reasoning Guidelines:
                    1. If there is a brand name or logo, you can reasoning with the help of the brand name or logo.
                    2. If there is no brand name or logo, you can specify the food or drink based on the visible characteristics.
                """),
            },
            "landmarks": {
                "reasoning_guide": textwrap.dedent("""
                Domain Reasoning Guidelines:
                    1. If it is a well known landmark like Eiffel Tower, you can directly specify the landmark based on the landmark name.
                    2. If it is not a well known building, you can identify the building based on the text in the image like the tower name, doorplate, or road name.
                """),
            },
            "text or chart reasoning": {
                "reasoning_guide": textwrap.dedent("""
                Domain Reasoning Guidelines:
                    1. In reasoning, carefully read the text or chart in the image and address the query step-by-step based on the text or chart. If you finally don't know the answer, you should say 'I don't know' in the reasoning.
                """),
            },
            "shopping product info": {
                "reasoning_guide": textwrap.dedent("""
                Domain Reasoning Guidelines:
                    1. If there is a brand name or logo, you can reasoning with the help of the brand name or logo.
                    2. If there is no brand name or logo, you can specify the product based on the visible characteristics.
                    3. Once you identify the product, if the query need additional knowledge that not in the image or your knowledge, clearly state "I don't know <specific missing information> about the <specific_product_name> in the image." in your reasoning.
                    4. If there is a hand, the queried item is probably the hold on the hand or pointed by the hand.
                """),
            },
            "vehicle": {
                "reasoning_guide": textwrap.dedent("""
                Domain Reasoning Guidelines:
                    1. You can identify the vehicle based on the brand name or logo and specific visible characteristics, like shape, windows, front, back, etc.
                    2. Once you identify the vehicle, if the query need additional knowledge that not in the image or your knowledge, clearly state "I don't know <specific missing information> about the <specific_vehicle_name> in the image." in your reasoning.
                """),
            },
            "other": { "reasoning_guide": "", "examples": ""}
        }
        return domain_system_prompt[domain]
    
    def _system_prompt(self, domain: str) -> str:
        # NOTE: ICL case is very important.
        # TODO: add reminder for year, time, price, etc.
        if domain not in self.info_list:
            raise ValueError(f"Domain '{domain}' is not supported. Supported domains are: {', '.join(self.info_list)}")
        return textwrap.dedent(f"""
                You are a visual assistant tasked with addressing the user's query for the image based on your inherent knowledge.
                General Reasoning Guidelines:
                    1. Generate step-by-step reasoning to address the query using evidence from the image and your knowledge. Limit your reasoning to no more than 5 concise steps, with each step written as a single sentence. Stop reasoning once you have enough information to answer the query or when you determine that necessary information is lacking.
                    2. In your reasoning, identify the exact object that the query is about by its exact name (e.g., car model, food name, brand name, species name, etc.). If no clear object matches the query, you may refer to textual clues visible in the image if available.
                    3. If the query involves multiple objects or relationships, dedicate one reasoning step to each object or relationship, and then summarize the result in a final step. 
                        For example:
                            1. The exact name of the object in the image that the query is about is <specific_object_name>. 
                            2. Next, the exact name of the object related to the first one is <specific_object_name>."
                            3. ...
                    4. If you cannot determine the necessary information from the image or the query, explicitly state: "I cannot determine the <what> that the query is about."
                    5. Do not suggest that the user to refer to external sources.
                    6. Always begin your reasoning with: "1. The exact name of the object that the query "<query>" is about is <specific_object_name>." and make your final reasoning concise. 
                    {self._domain_system_prompt(domain)["reasoning_guide"]}
                Output Format:
                    1. The exact name of the object in the image that the query is about is <specific_object_name>.
                    2. Then, I ...
                    ...
                    {{"reasoning": "<summary_reasoning_string>"}}
                
                Example 1:
                    User Query Image: 
                        An egocentric photo where a Christmas tree in the corner of the cluttered room.
                    User Query Prompt:
                        Given the <image>, please conduct step-by-step reasoning to address the query: which one of these in rockefeller square was taller, the 2023 one or the 2024 one?
                        Reasoning:
                    Assistant Output:
                        1. The exact name of the object in the image that the query "which one of these in rockefeller square was taller, the 2023 one or the 2024 one?" is about is <specific_object_name>.
                        2. There was <specific_object_name> at rockefeller square every year.
                        3. However, I don't know which one is taller because I don't have the height of the <specific_object_name> for 2023 and 2024.
                        Therefore, the answer is:
                        {{"reasoning": "I identified the <specific_object_name> in the image, but I don't know which one is taller since I don't have the height of the <specific_object_name> for 2023 and 2024."}}
                
                Example 2:
                    User Query Image: 
                        An egocentric photo of a car on the road.
                    User Query Prompt:
                        Given the <image>, please conduct step-by-step reasoning to address the query: can the 2023 model with a 1.3l awd engine of this vehicle get from washington, dc to baltimore on 5 gallons of gas?
                        Reasoning:
                    Assistant Output:
                        1. The exact name of the object in the image that the query "can the 2023 model with a 1.3l awd engine of this vehicle get from washington, dc to baltimore on 5 gallons of gas?" is about is <specific_object_name>.
                        2. The <specific_object_name> averages about 30 mpg.
                        3. With 5 gallons, it can travel approximately 150 miles.
                        4. This range easily covers the 40-mile trip from Washington, DC to Baltimore.
                        Therefore, the answer is:
                        {{"reasoning": "Yes, the <object_name> can get from Washington, DC to Baltimore on 5 gallons of gas."}}
                
                Example 3:
                    User Query Image:
                        A egocentric photo of a car on the road.
                    User Query Prompt:
                        Given the <image>, please conduct step-by-step reasoning to address the query: what is the price of the car?
                        Reasoning: 
                    Assistant Output:
                        1. Sorry, I cannot identify the car in the image.
                        2. Assuming I got the object's identity, the price is <visible/not visible> in the image and cannot be determined based on my knowledge.
                        Therefore, the answer is:
                        {{"reasoning": "I cannot determine the price of the car in the image. The price is <visible/not visible> in the image."}}
                {"Here is some domain examples: " if domain != "other" else ""}
                {self._domain_system_prompt(domain).get("examples", "")}
                """
            ).strip()
        
    def _prompt(self, query: str, domain: str):
        user_prompt = textwrap.dedent(f"""
                Given the <image>, please conduct step-by-step reasoning to address the query: {query}
                Reasoning:
                1. I identify the object that the query is about. It is a <object_name>.
                2. xxx
                3. xxx
                ...
                {{"reasoning": <your_reasoning_string>}}
                """
            ).strip()
        prompts = [
            {
                "role": "system",
                "content": self._system_prompt(domain)
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
    
    def __call__(self, images: List[Image.Image], queries: List[str], domains: Optional[List[str]] = None) -> List[str]:
        if domains is None or None in domains:
            domains = ["other"] * len(queries)
        return super().__call__(images=images, query=queries, domain=domains)


class ReasoningAgent(DomainAwareReasoningAgent):
    
    def __call__(self, images: List[Image.Image], queries: List[str], domains: Optional[List[str]] = None) -> List[str]:
        domains = ["other"] * len(queries)
        return super().__call__(images=images, queries=queries, domains=domains)

    def _prompt(self, query: str, domain: str = 'other'):
        return super()._prompt(query, 'other')


class DirectAnswerAgent(BaseVLLMAgent):
    def __init__(self, llm, tokenizer, generation_config, formater):
        super().__init__(llm, tokenizer, generation_config, formater)
        
    @property
    def _purpose(self):
        return "answer the user's question directly"
    
    @property
    def _schema_key(self):
        return "answer"
    
    @property
    def _value_type(self):
        return 'your_answer_string'
    
    def _system_prompt(self):
        return textwrap.dedent("""
                You are a helpful assistant that truthfully answers user questions about the provided image. If you are not sure about the answer, please say 'I don't know.'.
                """
            ).strip()
        
    def _prompt(self, query: str, msg_hist: List[Dict[str, Any]] | None = None):
        msg_hist_without_image = []
        if msg_hist is not None:
            for msg in msg_hist:
                if "<image>" in msg.get("content", ""):
                    continue
                msg_hist_without_image.append(msg)
        
        user_prompt = textwrap.dedent(f"""
                Given the <image> and query text: {query}
                Output your answer in JSON format like:
                {{"answer": <your_answer_string>}}
                """
            ).strip()
        
        prompts = [
            {
                "role": "system",
                "content": self._system_prompt()
            },
            {
                "role": "user",
                "content": [{"type": "image"}]
            }] + msg_hist_without_image + [
            {
                "role": "user",
                "content": user_prompt
            },
        ]
        return prompts
    
    def _output_postprocess(self, outputs: List[str]) -> List[str]:
        return [str(output) if output else "I don't know" for output in outputs]
    
    def __call__(
            self, 
            images: List[Image.Image], 
            queries: List[str],
            msg_hist: List[List[Dict[str, Any]] | None],
        ) -> List[str]:
        return super().__call__(images=images, query=queries, msg_hist=msg_hist)


class CoTAgent(BaseVLLMAgent):
    def __init__(self, llm, tokenizer, generation_config, formater):
        super().__init__(llm, tokenizer, generation_config, formater)
        
    @property
    def _purpose(self):
        return "conduct reasoning to answer the user's question based on the image"
    
    @property
    def _schema_key(self):
        return "answer"
    
    @property
    def _value_type(self):
        return 'your_answer_string'
    
    
    def _system_prompt(self) -> str:
        return textwrap.dedent(f"""
                You are a helpful assistant that truthfully answers user questions about the provided image step by step. If you are not sure about the answer, please say 'I don't know.'.
                Output Format:
                    step 1:
                    step 2:
                    ...
                    step n:
                    Final Answer:
                    {{"answer": "<final_answer_string>"}}
                """
            ).strip()
        
    def _prompt(self, query: str, msg_hist: List[Dict[str, Any]] | None = None):
        msg_hist_without_image = []
        if msg_hist is not None:
            for msg in msg_hist:
                if "<image>" in msg.get("content", ""):
                    continue
                msg_hist_without_image.append(msg)

        user_prompt = textwrap.dedent(f"""
                Given the <image> and query text: {query}
                Conduct step-by-step reasoning to answer the query and output your answer in JSON format like:
                step 1:
                step 2:
                ...
                step n:
                Final Answer:
                {{"answer": <your_answer_string>}}
                """
            ).strip()
        
        prompts = [
            {
                "role": "system",
                "content": self._system_prompt()
            },
            {
                "role": "user",
                "content": [{"type": "image"}]
            }] + msg_hist_without_image + [
            {
                "role": "user",
                "content": user_prompt
            },
        ]
        return prompts
    
    def _output_postprocess(self, outputs: List[str]) -> List[str]:
        return [str(output) if output else "I don't know" for output in outputs]
    
    def __call__(
            self, 
            images: List[Image.Image], 
            queries: List[str],
            msg_hist: List[List[Dict[str, Any]] | None] = None,
        ) -> List[str]:
        return super().__call__(images=images, query=queries, msg_hist=msg_hist)