import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Optional


class QwenReranker():
    def __init__(
            self, 
            model_path: str = "Qwen/Qwen3-Reranker-0.6B", 
            max_length: int = 8192, 
            instruction: Optional[str] = None
        ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                          torch_dtype=torch.float16, 
                                                          device_map="auto", 
                                                          low_cpu_mem_usage=True)
        self.model.eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        if instruction is None:
            self.instruction = 'Given a web search query, retrieve relevant passages that answer the query'

        self.max_length = max_length

    @staticmethod
    def _format_instruction(instruction: str, query: str, doc: str) -> str:
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
        return output
    
    def _process_inputs(self, pairs: List[str]) -> dict:
        inputs = self.tokenizer(pairs, padding=False, truncation='longest_first', return_attention_mask=False, 
                                max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, input_id in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + input_id + self.suffix_tokens

        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def _compute_logits(self, inputs: dict, **kwargs) -> torch.Tensor:
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp()
        return scores
    
    def _remove_list_duplicates(self, ori_list: List[str]) -> List[str]:
        seen = set()
        return [x for x in ori_list if not (x in seen or seen.add(x))]
    
    def rerank_batch(self, 
               questions: List[str], 
               entries: List[List[str]],
               weights: List[List[float]] = None,
               threshold: float = 0,
               batch_size: int = 32,
               ) -> List[Tuple[List[str], List[float]]]:
        return self.rerank(questions, entries, weights, threshold, batch_size)
               
    def rerank(self, 
               questions: List[str], 
               entries: List[List[str]],
               weights: List[List[float]] = None,
               threshold: float = 0,
               batch_size: int = 32,
               ) -> List[Tuple[List[str], List[float]]]:
    
        if weights is None:
            weights = [[1.0] * len(_entries) for _entries in entries]
        
        reranked_results = []
        for question, _entries, _weights in zip(questions, entries, weights, strict=True):
            if len(_entries) == 0:
                reranked_results.append(([], []))
                continue
            
            _entries = self._remove_list_duplicates(_entries)
            assert len(_entries) == len(_weights), "Entries and weights must have the same length."

            scores = []
            for i in range(0, len(_entries), batch_size):
                batch_entries = _entries[i:i + batch_size]
                pairs = [self._format_instruction(self.instruction, question, entry) for entry in batch_entries]
                inputs = self._process_inputs(pairs)
                batch_scores = self._compute_logits(inputs)
                scores.extend(batch_scores.tolist())
            
            scores = torch.tensor(scores)
            weighted_scores = scores * torch.tensor(_weights, dtype=torch.float32)
            weighted_scores, rerank_indices = weighted_scores.sort(descending=True)
            weighted_scores = weighted_scores[weighted_scores > threshold]
            rerank_indices = rerank_indices[:len(weighted_scores)]

            reranked_entries = [_entries[i] for i in rerank_indices]
            reranked_results.append((reranked_entries, weighted_scores.tolist()))
        return reranked_results


if __name__ == "__main__":
    # Example usage
    question = "What is the name of this store?"
    entries = ["This is a store that sells electronics.", 
               "Mount Fuji is an attractive volcanic cone. It has been a frequent subject of Japanese art, especially after 1600.",
               "Circle K Stores, Inc. is a Canadian-American chain of convenience stores headquartered in Tempe, Arizona."]
    
    reranker = QwenReranker()

    ## Test rerank
    reranked_results = reranker.rerank([question], [entries])
    print("Results for non-batch reranking:")
    for result in reranked_results:
        reranked_entries, scores = result
        for idx, (entry, score) in enumerate(zip(reranked_entries, scores, strict=True)):
            print(f"Rank {idx + 1}: {entry}, Score: {score:.4f}")

    # Test rerank_batch
    reranked_results = reranker.rerank([question] * 3, [entries] * 2 + [[]])
    print("\nResults for batch reranking:")
    for result in reranked_results:
        reranked_entries, scores = result
        for idx, (entry, score) in enumerate(zip(reranked_entries, scores, strict=True)):
            print(f"Rank {idx + 1}: {entry}, Score: {score:.4f}")
