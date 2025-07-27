from typing import List
from PIL import Image


class VLLMQueryDecomposer:

    def __init__(self, llm, tokenizer, config):
        self.llm = llm
        self.tokenizer = tokenizer
        self.config = config

    @property
    def system_prompt(self) -> str:
        return "You are a helpful assistant that breaks down complex image-based questions into simpler sub-questions."

    @staticmethod
    def decompose_prompt(query: str) -> str:
        return f"""Given the following question about an image, break it down into simpler sub-questions that will help figure out the answer of it.
Original question: {query}
Please provide at most 2-3 sub-questions in logical order that will help answer the main question. Format the sub-questions like:
- Question 1:
- Question 2:
...
"""

    def prompt(self, query: str, image: Image.Image) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": [{"type": "image"}] if image else []},
                {"role": "user", "content": self.decompose_prompt(query)},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )

    def decompose(self, query: str, image: Image.Image) -> List[str]:
        llm_input = {
            "prompt": self.prompt(query, image),
            "multi_modal_data": {"image": image} if image else {},
        }
        outputs = self.llm.generate(
            [llm_input],
            sampling_params=self.llm.sampling_params.__class__(
                temperature=self.config.tools.question_decomposition.config.llm_temperature,
                top_p=self.config.tools.question_decomposition.config.llm_top_p,
                max_tokens=self.config.tools.question_decomposition.config.max_tokens,
                skip_special_tokens=True,
            ),
        )
        response = outputs[0].outputs[0].text
        sub_queries = self._parse_response(response)
        return sub_queries

    def _parse_response(self, response: str) -> List[str]:
        sub_queries = list()
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                sub_query = line[line.find(":") + 1 :].strip()
                if sub_query:
                    sub_queries.append(sub_query)
        return sub_queries


class SmallLMQueryDecomposer:
    def __init__(self, llm, tokenizer, config):
        self.llm = llm
        self.tokenizer = tokenizer
        self.config = config

    def decompose(self, query: str, image: Image.Image) -> List[str]:
        raise NotImplementedError("small_lm mode is not implemented")


class SimpleQueryDecomposer:
    def __init__(self):
        pass

    def decompose(self, query: str, image: Image.Image) -> List[str]:
        return ["What is this", query]


class QueryDecomposer:
    def __init__(self, config, llm=None, tokenizer=None):
        self.config = config
        self.enable = self.config.tools.question_decomposition.enabled
        self.mode = self.config.tools.question_decomposition.config.mode
        self.llm = llm
        self.tokenizer = tokenizer
        self._decomposer = None
        if self.enable and self.config.enabled:
            if self.mode == "vllm":
                self._decomposer = VLLMQueryDecomposer(
                    self.llm, self.tokenizer, self.config
                )
            elif self.mode == "small_lm":
                self._decomposer = SmallLMQueryDecomposer(
                    self.llm, self.tokenizer, self.config
                )
            elif self.mode == "simple":
                self._decomposer = SimpleQueryDecomposer()
            else:
                raise ValueError(f"Unknown query decomposition type: {self.mode}")

    def decompose_queries(self, query: str, image: Image.Image) -> List[str]:
        # TODO: Parallelize this
        if self.enable and self.config.enabled:
            return self._decomposer.decompose(query, image)
        else:
            return [query]
