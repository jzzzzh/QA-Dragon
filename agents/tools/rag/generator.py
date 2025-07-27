import re
import logging
from typing import List, Dict, Any
from PIL import Image
from agents.modules.answer import BaseVLLMAgent

_ANSWER_SYS_PROMPT = "You are a helpful assistant that truthfully answers user questions about the provided image. If you are not sure about the answer, please say 'I don't know.'."

_ANSWER_USER_PROMPT = """Please answer the given question based on the provided image and your own knowledge.

Please think step by step and give a response containing the following parts:
- reason: the information from the image or your own knowledge that leads to the answer, which should be clear and concise within 2-3 sentences.
    - If you are not sure about the answer, please say 'I don't know.'.
- answer: your final answer in a concise format within one sentence.
    - The answer should include critical information from the reason to support your answer.
    - When referring to a specific object in the image, please use the name of the object, rather than 'this', 'that', or 'it'.
    - If the reason is 'I don't know.', please also say 'I don't know.' in the answer.

Required output format:
**Reason:** "your reason"
**Answer:** "your answer"

-Examples-
{examples}

-Real Data-
Question:
{question}

Your Response:
"""

_ANSWER_ICL_EXAMPLES = """Question: 
What is the model of this car?

Response:
**Reason:** Based on the image, the car is a Toyota Model AA, which was the first passenger car produced by Toyota.
**Answer:** The model of the car is Toyota Model AA.
"""


_ANSWER_PROMPT_WITH_CONTEXT = """Based on the provided image and retireved evidences, please answer the given question.
The evidences are some relevant information from the external knowledge base. Each of them is assigned a relevance score.
Note that the evidences are not always containing the correct answers, so you need combine them with your own knowledge to answer the question.

Please think step by step and give a response containing the following parts:
- reason: the information from the evidences or your own knowledge that leads to the answer, which should be clear and concise within 2-3 sentences.
    - If the evidences is not relevant to the question or the image, please say 'No relevant information.'.
    - If you cannot find the answer from the evidences, please say 'No relevant information.'.
    - If there are contradictory evidences, please say 'Contradictory evidences.'.
    - If you are not sure about the answer, please say 'I don't know.'.
- answer: your final answer in a concise format within one sentence. 
    - The answer should include critical information from the reason to support your answer.
    - When referring to a specific object in the image, please use the name of the object, rather than 'this', 'that', or 'it'.
    - When the reason is 'No relevant information.', 'Contradictory evidences.' or 'I don't know.', please also say 'I don't know.' in the answer.

Required output format:
**Reason:** "your reason" (2-3 sentences)
**Answer:** "your answer" (one sentence)

-Examples-
{examples}

-Real Data-
Context: 
{context}

Question: 
{question}

Your Response:
"""

_ANSWER_ICL_EXAMPLES_WITH_CONTEXT = """**Example 1**
Context:
Evidence 1 with relevance score 0.97:
We are proud to announce that our company (Axis) was awarded ISO 9001 version 2015 certification, a standard that provides all the requirements for a quality management system.
Evidence 2 with relevance score 0.87:
The ISO 9001 sets out the organizational requirements for a quality management system. That’s why Axis has taken the step of being certified, in particular for its recognition of quality.
Evidence 3 with relevance score 0.80:
ISO regularly updates its standards, and ISO 9001:2015 is the fifth edition released in September 2015.

Question:
which certification does this company have?

Response:
**Reason:** The evidence states that Axis was awarded ISO 9001 version 2015 certification for its quality management system.
**Answer:** Axis has ISO 9001 version 2015 certification for quality management.

**Example 2**
Context:
Evidence 1 with relevance score 0.95:
In 2022, it listed 2.16 million species on the planet, across a range of taxonomic groups — 1.05 million insects, over 11,000 birds, over 11,000 reptiles, and over 6,000 mammals. 
Evidence 2 with relevance score 0.90:
Currently, mycologists have described about 20,000 species around the globe, and the diversity among them is astounding. 

Question:
how many species of the genus betta?

Response:
**Reason:** No relevant information about the genus betta is provided in the evidences.
**Answer:** I don't know.

**Example 3**
Context: 
Evidence 1 with relevance score 1.00:
White wine has a lower calorie count, with 205 calories per glass at the low end and 205 calories per glass at the high end.
Evidence 2 with relevance score 0.85:
Comparatively, white wine calories hover around an average of 24 per ounce or around 600 calories per 750mL bottle, so only slightly fewer calories than red wines.

Question: 
does this have less calories than red wine?

Response:
**Reason:** The evidence states that calories in white wine hover around an average of 24 per ounce, which is lower than the average calories in red wine.
**Answer:** Yes, on average white wine has fewer calories than red wine.

**Example 4**
Context:
Evidence 1 with relevance score 0.95:
Red Foxes were introduced to Australia in the 1870s. By the 1890s they had become widespread feral pests. 
Evidence 2 with relevance score 0.80:
Red Foxes became established in Australia through successive introductions by settlers in 1830s.

Question:
when were red foxes introduced to Australia?

Response:
**Reason:** Contradictory evidences are provided in the evidences. One evidence states that Red Foxes were introduced in the 1870s, while another states that they were introduced in the 1830s.
**Answer:** I don't know.
"""

class AnswerGenerator(BaseVLLMAgent):
    NEGATIVE_RESPONSES = [
        "no relevant information",
        "contradictory evidence",
        "i don't know",
    ]

    IDK_RESPONSE = "I don't know."

    def __init__(self, llm, tokenizer, config):
        generation_config = config.config

        self.return_reason = generation_config.get("return_reason", False)
        self.whitebox_detection = generation_config.get("whitebox_detection", False)
        self.norm_prob_threshold = generation_config.get("norm_prob_threshold", 0.7)
        self.min_prob_threshold = generation_config.get("min_prob_threshold", 0.1)

        if self.whitebox_detection:
            assert generation_config.get("logprobs", None) == 1
            logging.warning(
                f"Using whitebox detection with norm_prob_threshold={self.norm_prob_threshold} "
                f"and min_prob_threshold={self.min_prob_threshold}.\n "
                f"Current temperature is {generation_config['temperature']}")
        
        self.sys_prompt = _ANSWER_SYS_PROMPT
        self.user_prompt = _ANSWER_USER_PROMPT.format(
            examples=_ANSWER_ICL_EXAMPLES,
            question="{question}"
        )
        self.user_prompt_with_context = _ANSWER_PROMPT_WITH_CONTEXT.format(
            examples=_ANSWER_ICL_EXAMPLES_WITH_CONTEXT,
            context="{context}",
            question="{question}"
        )

        self.answer_pattern = r"\*\*Answer:\*\*(.*?)(?=\*\*|\n|\Z)"
        self.reason_pattern = r"\*\*Reason:\*\*(.*?)(?=\*\*Answer:\*\*|\n|\Z)"

        self.register = {}

        super().__init__(llm, tokenizer, generation_config, formater=None)

    @property
    def _system_prompt(self):
        return self.sys_prompt

    def _prompt(self, queries: str, rag_ctx: str | None, msg_hist: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
        msg_hist_without_image = []
        if msg_hist is not None:
            for msg in msg_hist:
                if "<image>" in msg.get("content", ""):
                    continue
                msg_hist_without_image.append(msg)

        if rag_ctx is not None and rag_ctx != "":
            user_prompt = self.user_prompt_with_context.format(context=rag_ctx, question=queries)
        else:
            user_prompt = self.user_prompt.format(question=queries)
        
        prompts = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": [{"type": "image"}]}
        ] + msg_hist_without_image + [
            {"role": "user", "content": user_prompt}
        ]

        if self.register.get("prompts"):
            self.register["prompts"].append(prompts)
        return prompts
    
    def _output_postprocess(self, outputs: List[Dict[str, Any]]):
        """
        Parse the outputs from the model to extract the reason and answer.
        The response should be in the format:
        **Reason:** "your reason"
        **Answer:** "your answer"
        """
        reasons, answers = [], []
        for output in outputs:
            text_output = output.text

            reason_match = re.search(self.reason_pattern, text_output, re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else ""
            if reason == "":
                reason = f"<no reason parsed> {self.IDK_RESPONSE}"

            answer_match = re.search(self.answer_pattern, text_output, re.DOTALL)
            answer = answer_match.group(1).strip() if answer_match else ""
            if answer == "":
                reason = f"<no answer parsed> {reason}"
                answer = self.IDK_RESPONSE
            
            if any(resp.lower() in reason.lower() for resp in self.NEGATIVE_RESPONSES):
                reason = f"<negative response in reason> {reason}"
                answer = self.IDK_RESPONSE
            
            if any(resp.lower() in answer.lower() for resp in self.NEGATIVE_RESPONSES):
                reason = f"<negative response in answer> {reason}"
                answer = self.IDK_RESPONSE

            if self.whitebox_detection:
                # Calculate the certainty score based on the logprobs
                certainty_score = self._calculate_certainty(output)
                norm_prob, min_prob = certainty_score["norm_prob"], certainty_score["min_prob"]
                norm_prob, min_prob = round(norm_prob, 4), round(min_prob, 4)

                reason = f"<norm_prob: {norm_prob}, min_prob: {min_prob}> {reason}"
                if (norm_prob < self.norm_prob_threshold) or (min_prob < self.min_prob_threshold):
                    reason = f"<low certainty> {reason}"
                    answer = self.IDK_RESPONSE
                    logging.warning(f"Low certainty in generator: norm_prob={norm_prob}, min_prob={min_prob}")
                else:
                    reason = f"<high certainty> {reason}"

                self.register["norm_probs"].append(norm_prob)
                self.register["min_probs"].append(min_prob)

            reasons.append(reason)
            answers.append(answer)
        
        return reasons, answers

    def __call__(
            self,
            queries: List[str],
            images: List[Image.Image],
            rag_ctx: List[str | None],
            msg_hist: List[List[Dict[str, Any]] | None],
            return_prompts: bool = False
        ) -> Dict[str, Any]:

        if return_prompts:
            self.register["prompts"] = []

        if self.whitebox_detection:
            self.register["norm_probs"] = []
            self.register["min_probs"] = []
        
        (reasons, answers), raw_answers = super().__call__(
            images=images, 
            queries=queries,
            rag_ctx=rag_ctx,
            msg_hist=msg_hist,
            use_ori_outputs=True
        )
        
        results = {"answers": answers, "reasons": reasons}
        if return_prompts:
            results["prompts"] = self.register.pop("prompts")
        
        return results, raw_answers
