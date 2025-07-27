import re
import logging
import vllm

from typing import List, Dict, Any
from PIL import Image
from agents.modules.answer import BaseVLLMAgent


_VERIFIER_SYS_PROMPT = """You are a helpful assistant that evaluates whether the agent's answer to the user's image query is reasonable based on the evidence."""

_VERIFIER_EVIDENCE_PROMPT = """Given an image query, the retrieved evidences, and the agent's candidate answer, please check the correctness of the answer.
1. If you cannot find the information in the image and evidences to support the provided answer, please respond with 'Response: Incorrect Answer'
2. If there are contradictory evidences, please respond with 'Response: Incorrect Answer'
3. If you think the candidate answer is reasonable based on the image and evidences, please respond with 'Response: Correct Answer'
4. If you cannot determine whether the answer is correct or incorrect, please respond with 'Response: Uncertain'

-Real Input-
Query: {question}\n
Evidence: {evidence}\n
Answer: {answer}

-Your output (choose from 'Incorrect Answer', 'Correct Answer', or 'Uncertain' without any further explanation)-
Response: 
"""

_VERIFIER_EVIDENCE_PROMPT_V2 = """Given an image query, the retrieved evidences, and the agent’s candidate answer, assess the correctness of the answer by following these guidelines:
1. Unsupported Answer. If the answer is not supported by the image and evidences, please respond the following:
    **Reason:** Briefly (1–2 sentences) explain why the answer lacks sufficient support.
	**Response:** Incorrect Answer
2. Contradicted Evidence. If there are conflicting information in the evidences, please respond the following:
    **Reason:** Briefly (1–2 sentences) state the specific contradictory evidences.
	**Response:** Incorrect Answer
3. Unclear or Incomplete Answer. If the answer is vague or fails to fully address the question, please respond the following:
	**Reason:** Briefly (1–2 sentences) explain why the answer is unclear or incomplete.
	**Response:** Incorrect Answer
4. Correct Answer. If the answer is fully supported by the image and evidences, please respond the following:
	**Reason:** Briefly (1–2 sentences) state why the answer is correct.
	**Response:** Correct Answer

Required output format:
**Reason:** "your explanation"  
**Response:** "Correct Answer" or "Incorrect Answer"

-Real Input-
Query: {question}\n
Evidence: {evidence}\n
Answer: {answer}

Your Output:
"""

_VERIFIER_REASON_PROMPT = """Given an image query and the agent's reasoning evidence and candidate answer, please check the correctness of the answer.
1. If you find the reasoning evidence does not support the candidate answer, please respond with 'Response: Incorrect Answer'
2. If you think the candidate answer is reasonable based on the image and reasoning evidence, please respond with 'Response: Correct Answer'
3. If you cannot determine whether the answer is correct or incorrect, please respond with 'Response: Uncertain'

-Real Input-
Query: {question}\n
Evidence: {evidence}\n
Answer: {answer}

-Your output (choose from 'Incorrect Answer', 'Correct Answer', or 'Uncertain' without any further explanation)-
Response: 
"""

_VERIFIER_EVIDENCE_EXAMPLES = """Example 1:
-Input-
Query: What is the price of the car?
Evidence: 
Evidence 1 with relevance score 0.31:
At the onset of World War II, Toyota almost exclusively produced standard-sized trucks for the Japanese Army, which paid one-fifth of the price in advance and the remainder in cash upon delivery.
Evidence 2 with relevance score 0.29:
In April 1936, Toyoda's first passenger car (the Model AA) was completed, significantly cheaper than Ford or GM cars.
Evidence 3 with relevance score 0.10:
In the 1960s, Toyota took advantage of the rapidly growing Japanese economy to sell cars to a growing middle-class, leading to the development of the Toyota Corolla, which became the world's all-time best-selling automobile.
Answer: The car is priced at $20,000.

-Output-
Response: Incorrect Answer

Example 2:
-Input-
Query: what is this made out of?
Evidence: 
Evidence 1 with relevance score 0.24:
daiya cream cheese, dairy free, plant based, original 8 oz | cream cheese | super saver you& x27 ve never tasted dairy free cream cheese like this before now made with daiya oat cream blend, our new-and-improved plant based flavors are smooth, creamy, and dreamier than ever try dairy free
Evidence 2 with relevance score 0.20:
daiya reformulates alternative cheese line | dairy processing the company invested in fermentation technology to develop an oat cream for its products ingredients and formulating innovations product development trends cultures/enzymes free-from plant-based new products alt-dairy cheese dairy alternatives plant-based \u00b7 keywords alternative dairy cheese daiya foods fermentation flavor food science
Evidence 3 with relevance score 0.17:
daiya cheese | best dairy free mac & cheese, sauces, pizza & more leading a vegan lifestyle is so much easier when you\u2019ve got daiya welcome to the world of daiya cheeses, flatbread, pizzas, and mac and cheese daiya\u2019s cheese, cream cheese, and mac and cheese are dairy-dodging reinventions of"
Answer: This product is made out of oat cream.

-Output-
Response: Correct Answer
"""

# TODO: Add examples for reason-based verification
_VERIFIER_REASON_EXAMPLES = ""


class Verifier(BaseVLLMAgent):
    NEGATIVE_RESPONSES = [
        " incorrect",
        "incorrect ",
        "uncertain",
    ]
    POSITIVE_RESPONSES = [
        " correct",
        "correct ",
    ]

    IDK_RESPONSE = "I don't know."

    def __init__(self, llm, tokenizer, config):
        self.enabled = config.enabled
        self.config = config.config
        self.use_reason = self.config.get("use_reason", False)
        self.reject_no_evidence = self.config.get("reject_no_evidence", True)
        self.whitebox_detection = self.config.get("whitebox_detection", False)
        self.norm_prob_threshold = self.config.get("norm_prob_threshold", 0.7)
        self.min_prob_threshold = self.config.get("min_prob_threshold", 0.1)

        if self.enabled:
            logging.warning(f"Verifier is enabled with config: {self.config}")

            if self.use_reason:
                raise ValueError("Reason-based verification is not implemented yet.")
            
            if self.reject_no_evidence:
                logging.warning("Rejecting queries without evidence is enabled.")

            if self.whitebox_detection:
                assert self.config.get("logprobs", None) == 1
                logging.warning(
                    f"Using whitebox detection in verifier with norm_prob_threshold={self.norm_prob_threshold} "
                    f"and min_prob_threshold={self.min_prob_threshold}. "
                    f"Current temperature is {self.config['temperature']}")
        else:
            logging.warning("Verifier is disabled, using the default answer as the output.")

        self.sys_prompt = _VERIFIER_SYS_PROMPT

        if self.use_reason:
            self.user_prompt = _VERIFIER_REASON_PROMPT
            examples = _VERIFIER_REASON_EXAMPLES
        else:
            self.user_prompt = _VERIFIER_EVIDENCE_PROMPT_V2
            examples = _VERIFIER_EVIDENCE_EXAMPLES

        if "{examples}" in self.user_prompt:
            self.user_prompt = self.user_prompt.format(
                examples=examples,
                question="{question}",
                evidence="{evidence}",
                answer="{answer}",
            )
        
        self.reason_pattern = r"\*\*Reason:\*\*(.*?)(?=\*\*Response:\*\*|\n|\Z)"
        self.response_pattern = r"\*\*Response:\*\*(.*?)(?=\*\*|\n|\Z)"

        self.register = {}
        super().__init__(llm, tokenizer, self.config, formater=None)

    @property
    def _system_prompt(self):
        return self.sys_prompt

    def _prompt(self, queries: str, evidences: str, answers: str, reasons: str=None) -> List[Dict[str, Any]]:
        if reasons:
            answers = f"{answers} \n Reason from the evidence: {reasons}"
        
        user_prompt = self.user_prompt.format(
            question=queries, 
            evidence=evidences, 
            answer=answers
        )

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
    
    def _output_postprocess(self, outputs: List[str]):
        answers = self.register.pop("answers")
        evidences = self.register.pop("evidences")
        
        rejects = []
        final_answers = []
        if self.enabled:
            for output, answer, evidence in zip(outputs, answers, evidences, strict=True):
                text_output = output.text
                response_match = re.search(self.response_pattern, text_output, re.IGNORECASE)
                
                if response_match:
                    response = response_match.group(1).strip().lower()
                    if any(neg_response in response for neg_response in self.NEGATIVE_RESPONSES):
                        reject = "<negative response in verifier>"
                        final_answer = self.IDK_RESPONSE
                    elif any(pos_response in response for pos_response in self.POSITIVE_RESPONSES):
                        reject = False
                        final_answer = answer
                    else:
                        reject = "<error response in verifier>"
                        final_answer = self.IDK_RESPONSE
                else:
                    reject = "<error response in verifier>"
                    final_answer = self.IDK_RESPONSE
                
                if self.reject_no_evidence and not evidence:
                    reject ="<no evidence provided>"
                    final_answer = self.IDK_RESPONSE
                
                if self.whitebox_detection:
                    # Calculate the certainty score based on the logprobs
                    certainty_score = self._calculate_certainty(output)
                    norm_prob, min_prob = certainty_score["norm_prob"], certainty_score["min_prob"]
                    norm_prob, min_prob = round(norm_prob, 4), round(min_prob, 4)

                    if (norm_prob < self.norm_prob_threshold) or (min_prob < self.min_prob_threshold):
                        reject = f"<low certainty in verifier> <norm_prob: {norm_prob}, min_prob: {min_prob}>"
                        final_answer = self.IDK_RESPONSE
                        logging.warning(f"Low certainty in verifier: norm_prob={norm_prob}, min_prob={min_prob}")

                rejects.append(reject)
                final_answers.append(final_answer)
        else:
            final_answers = answers
            rejects = [False] * len(answers)
        
        self.register["rejects"] = rejects
        return final_answers
    
    def __call__(self, 
        images: List[Image.Image | None],
        queries: List[str],
        evidences: List[str],
        answers: List[str],
        reasons: List[str] = None
        ):

        self.register["answers"] = answers
        self.register["evidences"] = evidences

        evidences = [evidence if evidence else "No evidence provided." for evidence in evidences]
        reasons = [None] * len(queries) if reasons is None else reasons
        
        outputs, raw_outputs = super().__call__(
            images=images, 
            queries=queries, 
            evidences=evidences, 
            answers=answers, 
            reasons=reasons,
            use_ori_outputs=True
        )

        rejects = self.register["rejects"]
        for i, reject in enumerate(rejects):
            if reject:
                raw_outputs[i] = f"{reject} {raw_outputs[i]}"

        return outputs, raw_outputs
    
    # def __call__(self, 
    #     images: List[Image.Image | None],
    #     queries: List[str],
    #     evidences: List[str],
    #     answers: List[str],
    #     reasons: List[str] = None
    #     ):
        
    #     if self.enabled:
    #         self.register["answers"] = answers
    #         self.register["evidences"] = evidences

    #         evidences = [evidence if evidence else "No evidence provided." for evidence in evidences]
    #         reasons = [None] * len(queries) if reasons is None else reasons
            
    #         outputs, raw_outputs = super().__call__(
    #             images=images, 
    #             queries=queries, 
    #             evidences=evidences, 
    #             answers=answers, 
    #             reasons=reasons,
    #             use_ori_outputs=True
    #         )

    #         rejects = self.register["rejects"]
    #         for i, reject in enumerate(rejects):
    #             if reject:
    #                 raw_outputs[i] = f"{reject} {raw_outputs[i]}"
    #     else:
    #         outputs = answers
    #         raw_outputs = ["<verifier disabled>"] * len(answers)
    #         self.register["rejects"] = [False] * len(answers)

    #     return outputs, raw_outputs


if __name__ == "__main__":
    import json
    from omegaconf import OmegaConf
    from glob import glob
    from tqdm import tqdm
    from PIL import Image
    from agents.utils.utils import resize_image

    cfg = OmegaConf.load("configs/basic.yaml")
    verifier_config = cfg.postprocessing.verifier

    llm = vllm.LLM(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        tensor_parallel_size=cfg.basic.VLLM_TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=cfg.basic.VLLM_GPU_MEMORY_UTILIZATION,
        max_model_len=cfg.basic.MAX_MODEL_LEN,
        max_num_seqs=cfg.basic.MAX_NUM_SEQS,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
    )
    tokenizer = llm.get_tokenizer()

    # Initialize Verifier
    verifier = Verifier(llm, tokenizer, verifier_config)

    result_dir = cfg.saver.config.save_path
    golden_answers = json.load(open(f"{result_dir}/output.json"))
    golden_answers = {str(data["idx"]): data["gold"] for data in golden_answers }

    verified_results = []
    batch_dirs = sorted(glob(f"{result_dir}/batch_*"))
    for batch_dir in tqdm(batch_dirs):
        query_dirs = sorted(glob(f"{batch_dir}/query_*"))
        for query_dir in query_dirs:
            
            query_id = query_dir.split("/")[-1].split("_")[-1]
            query = json.load(open(f"{query_dir}/query.json"))
            answer =json.load(open(f"{query_dir}/answer.json"))
            evidence = json.load(open(f"{query_dir}/rag_context.json"))
            image = Image.open(f"{query_dir}/image.jpg")
            image = resize_image(image)

            verified_answer = verifier(
                images=[image],
                queries=[query],
                evidences=[evidence],
                answers=[answer],
            )[0]

            verified_results.append({
                "query_id": query_id,
                "query": query,
                "answer": answer,
                "verified_answer": verified_answer,
                "gold_answer": golden_answers[query_id],
            })
    
    with open(f"{result_dir}/verified_results.json", "w") as f:
        json.dump(verified_results, f, indent=4)
