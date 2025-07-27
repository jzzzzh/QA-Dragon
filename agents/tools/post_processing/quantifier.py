import pandas as pd

from typing import Any, List, Optional, Dict
from PIL import Image
from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier
from uqlm.black_box import BertScorer, CosineScorer, MatchScorer
from agents.modules.base import BaseVLLMAgent


class VLLMUncertaintyQuantifier(UncertaintyQuantifier):
    def __init__(
        self,
        agent: BaseVLLMAgent = None,
        device: str = "cuda",
        postprocessor: Optional[Any] = None,
    ) -> None:
        """
        Parent class for uncertainty quantification of VLLM responses.
        Parameters
        ----------
        agent : BaseVLLMAgent
            A VLLM agent object to get passed to the quantifier. 
        postprocessor : callable, default=None
            A callable function that takes a list of responses and applies postprocessing to them.
            If None, no postprocessing is applied.
        """
        super().__init__(device=device, postprocessor=postprocessor)
        self.agent = agent

    def generate_original_responses(
        self, 
        images: Optional[List[Image.Image]] = None, 
        **prompt_kwargs
    ) -> List[str]:
        """
        This method generates original responses for uncertainty
        estimation. If specified in the child class, all responses are postprocessed
        using the callable function defined by the user.
        """
        print("Generating responses...")
        responses = self._generate_responses(count=1, images=images, **prompt_kwargs)
        responses = [r[0] for r in responses]  # Flatten the list
        return responses
    
    def generate_candidate_responses(
        self, 
        images: Optional[List[Image.Image]] = None, 
        **prompt_kwargs
    ) -> List[List[str]]:
        """
        This method generates multiple responses for uncertainty
        estimation. If specified in the child class, all responses are postprocessed
        using the callable function defined by the user.
        """
        print("Generating candidate responses...")
        sampled_responses = self._generate_responses(count=self.num_responses, temperature=self.sampling_temperature,
                                                     images=images, **prompt_kwargs)
        return sampled_responses
    
    def _generate_responses(
        self,
        temperature: float = None,
        count: int = 1,
        images: Optional[List[Image.Image]] = None, 
        **prompt_kwargs
    ) -> List[List[str]]:
        """Helper function to generate responses with LLM"""
        if self.agent is None:
            raise ValueError(
                """llm must be provided to generate responses."""
            )
        temperature_backup = self.agent.generation_config['temperature']
        if temperature:
            self.agent.generation_config['temperature'] = temperature
        
        batch_size = len(next(iter(prompt_kwargs.values())))

        # [num_prompts, count]
        responses_list = [[] for _ in range(batch_size)]
        for _ in range(count):
            responses, _ = self.agent(images=images, **prompt_kwargs)
            if self.postprocessor:
                responses = self.postprocessor(responses)

            for i in range(batch_size):
                responses_list[i].append(responses[i])

        self.agent.generation_config['temperature'] = temperature_backup
        return responses_list
    
    def _construct_judge(self, llm: Any = None):
        raise NotImplementedError("This method is not supported in VLLMUncertaintyQuantifier.")


class UQResult:
    def __init__(self, result: Dict[str, Any]) -> None:
        """
        Class that characterizes result of an UncertaintyQuantifier.

        Parameters
        ----------
        result: dict
            A dictionary that is defined during `evaluate` or `tune_params` method
        """
        self.data = result.get("data")
        self.metadata = result.get("metadata")
        self.parameters = result.get("parameters")
        self.confidence_scores = self.data.get("confidence_scores")
        self.responses = self.data.get("responses")
        self.sampled_responses = (
            None if not self.data.get("sampled_responses") else self.data.get("sampled_responses")
        )
        self.result_dict = result

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns result in dictionary form
        """
        return self.result_dict

    def to_df(self) -> pd.DataFrame:
        """
        Returns result in pd.DataFrame
        """
        rename_dict = {
            col: col[:-1]
            for col in self.result_dict["data"].keys()
            if col.endswith("s") and col != "sampled_responses"
        }

        return pd.DataFrame(self.result_dict["data"]).rename(columns=rename_dict)

    
class VLLMBlackBoxUQ(VLLMUncertaintyQuantifier):
    def __init__(
        self,
        agent: BaseVLLMAgent= None,
        device: str = "cuda",
        scorers: Optional[List[str]] = None,
        use_best: bool = True,
        nli_model_name: str = "microsoft/deberta-large-mnli",
        postprocessor: Any = None,
        max_length: int = 2000,
        sampling_temperature: float = 1.0,
        verbose: bool = False,
    ) -> None:
        super().__init__(agent=agent, device=device, postprocessor=postprocessor)

        self.prompts = None
        self.max_length = max_length
        self.verbose = verbose
        self.use_best = use_best
        self.sampling_temperature = sampling_temperature
        self.nli_model_name = nli_model_name
        self._validate_scorers(scorers)
        self.use_nli = ("semantic_negentropy" in self.scorers) or (
            "noncontradiction" in self.scorers
        )
        if self.use_nli:
            self._setup_nli(nli_model_name)

    def generate_and_score(
        self,
        num_responses: int = 5,
        original_responses: Optional[List[str]] = None,
        images: Optional[List[Image.Image]] = None, 
        **prompt_kwargs
    ) -> UQResult:
        """
        Generate LLM responses, sampled LLM (candidate) responses, and compute confidence scores with specified scorers for the provided prompts.

        Returns
        -------
        UQResult
            UQResult containing data (prompts, responses, and scores) and metadata
        """
        self.num_responses = num_responses

        if original_responses is not None:
            responses = original_responses
        else:
            responses = self.generate_original_responses(images, **prompt_kwargs)
        sampled_responses = self.generate_candidate_responses(images, **prompt_kwargs)
        # NOTE: skip the "I don't know" in sampled_responses because it is caused by the exception and should not be used for scoring
        sampled_responses = [[response for response in sampled_response if "i don't know" not in response.lower()] or ["I don't know"] * len(sampled_response) for sampled_response in sampled_responses]
        return self.score(
            responses = responses, sampled_responses = sampled_responses,
        )

    def score(
        self, responses: List[str], sampled_responses: List[List[str]]
    ) -> UQResult:
        """
        Compute confidence scores with specified scorers on provided LLM responses. Should only be used if responses and sampled responses
        are already generated. Otherwise, use `generate_and_score`.

        Parameters
        ----------
        responses : list of str, default=None
            A list of model responses for the prompts. 

        sampled_responses : list of list of str, default=None
            A list of lists of sampled LLM responses for each prompt. These will be used to compute consistency scores by comparing to 
            the corresponding response from `responses`.

        Returns
        -------
        UQResult
            UQResult containing data (prompts, responses, and scores) and metadata
        """
        print("Computing confidence scores...")
        self.responses = responses
        self.sampled_responses = sampled_responses
        self.num_responses = len(sampled_responses[0])
        
        self.scores_dict = {k: [] for k in self.scorer_objects}
        if self.use_nli:
            compute_entropy = ("semantic_negentropy" in self.scorers)
            nli_scores = self.nli_scorer.evaluate(
                responses=self.responses,
                sampled_responses=self.sampled_responses,
                use_best=self.use_best,
                compute_entropy=compute_entropy,
            )
            if self.use_best:
                self.original_responses = self.responses.copy()
                self.responses = nli_scores["responses"]
                self.sampled_responses = nli_scores["sampled_responses"]

            for key in ["semantic_negentropy", "noncontradiction"]:
                if key in self.scorers:
                    if key == "semantic_negentropy":
                        nli_scores[key] = [
                            1 - s
                            for s in self.nli_scorer._normalize_entropy(
                                nli_scores[key]
                            )
                        ]  # Convert to confidence score
                    self.scores_dict[key] = nli_scores[key]

        # similarity scorers that follow the same pattern
        for scorer_key in ["exact_match", "bert_score", "bleurt", "cosine_sim"]:
            if scorer_key in self.scorer_objects:
                self.scores_dict[scorer_key] = self.scorer_objects[scorer_key].evaluate(
                    responses=self.responses,
                    sampled_responses=self.sampled_responses,
                )
                
        return self._construct_result()
                
    def _construct_result(self) -> Any:
        """Constructs UQResult object"""
        data = {
            "responses": self.responses,
            "sampled_responses": self.sampled_responses,
        }
        data.update(self.scores_dict)
        result = {
            "data": data,
            "metadata": {
                "temperature": None if not self.agent else self.agent.generation_config['temperature'],
                "sampling_temperature": None if not self.sampling_temperature else self.sampling_temperature,
                "num_responses": self.num_responses,
                "scorers": self.scorers,
            },
        }
        return UQResult(result)

    def _validate_scorers(self, scorers: List[Any]) -> None:
        "Validate scorers and construct applicable scorer attributes"
        self.scorer_objects = {}
        if scorers is None:
            scorers = self.default_black_box_names
        for scorer in scorers:
            if scorer == "exact_match":
                self.scorer_objects["exact_match"] = MatchScorer()
            elif scorer == "bert_score":
                self.scorer_objects["bert_score"] = BertScorer()
            elif scorer == "bleurt":
                from uqlm.black_box import BLEURTScorer
                self.scorer_objects["bleurt"] = BLEURTScorer()
            elif scorer == "cosine_sim":
                self.scorer_objects["cosine_sim"] = CosineScorer()
            elif scorer in ["semantic_negentropy", "noncontradiction"]:
                continue
            else:
                raise ValueError(
                    """
                    scorers must be one of ['semantic_negentropy', 'noncontradiction', 'exact_match', 'bert_score', 'bleurt', 'cosine_sim']
                    """
                )
        self.scorers = scorers
