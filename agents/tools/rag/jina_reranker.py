import torch
from PIL import Image
from transformers import AutoModel
from typing import List, Tuple, Optional


class JinaReranker():
    def __init__(
            self, 
            model_path: str = "jinaai/jina-reranker-m0", 
            max_length: int = 8192, 
            device: str = 'cuda'
        ) -> None:
        
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype="auto",
            trust_remote_code=True,
            # attn_implementation="flash_attention_2"
        )
        self.model.to(device)
        self.model.eval()
        self.max_length = max_length
    
    def _remove_list_duplicates(self, ori_list: List[str]) -> List[str]:
        seen = set()
        return [x for x in ori_list if not (x in seen or seen.add(x))]
    
    def rerank_batch(self, 
               questions: List[str], 
               images: List[Image.Image], 
               entries: List[List[str]],
               weights: List[List[float]] = None,
               threshold: float = 0,
               batch_size: int = 32,
               ) -> List[Tuple[List[str], List[float]]]:
        return self.rerank(questions, images, entries, weights, threshold, batch_size)
               
    def rerank(self, 
               questions: List[str], 
               images: List[Image.Image], 
               entries: List[List[str]],
               weights: List[List[float]] = None,
               threshold: float = 0,
               batch_size: int = 32,
               ) -> List[Tuple[List[str], List[float]]]:
    
        if weights is None:
            weights = [[1.0] * len(_entries) for _entries in entries]
        
        reranked_results = []
        for question, image, _entries, _weights in zip(questions, images, entries, weights, strict=True):
            if len(_entries) == 0:
                reranked_results.append(([], []))
                continue
            
            _entries = self._remove_list_duplicates(_entries)
            assert len(_entries) == len(_weights), "Entries and weights must have the same length."

            scores_with_question, scores_with_image = [], []
            for i in range(0, len(_entries), batch_size):
                batch_entries = _entries[i:i + batch_size]
                
                # calculate scores with question
                pairs = [[question, doc] for doc in batch_entries]
                batch_scores = self.model.compute_score(pairs, max_length=self.max_length, query_type="text", doc_type="text")
                scores_with_question.extend(batch_scores.tolist())

                # calculate scores with image
                pairs = [[image, doc] for doc in batch_entries]
                batch_scores = self.model.compute_score(pairs, max_length=self.max_length, query_type="image", doc_type="text")
                scores_with_image.extend(batch_scores.tolist())
            
            scores_with_question = torch.tensor(scores_with_question)
            scores_with_image = torch.tensor(scores_with_image)
            scores = scores_with_question + scores_with_image / 2.0

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
    image = Image.open("./docs/dataset_info/multiqa.jpg")
    entries = ["This is a store that sells electronics.", 
               "Mount Fuji is an attractive volcanic cone. It has been a frequent subject of Japanese art, especially after 1600.",
               "Circle K Stores, Inc. is a Canadian-American chain of convenience stores headquartered in Tempe, Arizona."]
    
    reranker = JinaReranker()

    ## Test rerank
    reranked_results = reranker.rerank([question], [image], [entries])
    print("Results for non-batch reranking:")
    for result in reranked_results:
        reranked_entries, scores = result
        for idx, (entry, score) in enumerate(zip(reranked_entries, scores, strict=True)):
            print(f"Rank {idx + 1}: {entry}, Score: {score:.4f}")

    # Test rerank_batch
    reranked_results = reranker.rerank([question] * 3, [image] * 3, [entries] * 2 + [[]])
    print("\nResults for batch reranking:")
    for result in reranked_results:
        reranked_entries, scores = result
        for idx, (entry, score) in enumerate(zip(reranked_entries, scores, strict=True)):
            print(f"Rank {idx + 1}: {entry}, Score: {score:.4f}")
