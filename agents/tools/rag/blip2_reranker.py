import torch
from PIL import Image
from typing import List, Tuple
from agents.lavis.models import load_model_and_preprocess


class Blip2Reranker():
    def __init__(self, model_name="blip2_reranker", model_type="reranker", dtype=torch.float16):
        self.dtype = dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        blip_model, vis_processors, txt_processors = load_model_and_preprocess(
            name=model_name, model_type=model_type, is_eval=True, device=self.device
        )

        self.txt_processor = txt_processors["eval"]
        self.vis_processor = vis_processors["eval"]

        self.blip_model = blip_model
        self.blip_model.use_vanilla_qformer = True
        self.blip_model.to(self.dtype)

    def _remove_list_duplicates(self, ori_list: List[str]) -> List[str]:
        seen = set()
        return [x for x in ori_list if not (x in seen or seen.add(x))]
    
    def rerank_batch(self, 
        questions: List[str], 
        images: List[Image.Image], 
        entries: List[List[str]],
        threshold: float = 0,
        batch_size: int = 32,
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Rerank the retrieved entries based on their similarity scores.
        Args:
            questions (list): List of questions asked.
            images (list): List of images associated with the questions.
            entries (list): List of retrieved entries.
        Returns:
            list: A list of tuples, each containing the reranked entries and their similarity scores.
        """
        assert len(questions) == len(images) == len(entries), "Questions, images, and entries must have the same length."
        
        images = [self.vis_processor(image) for image in images]
        images = torch.stack(images, dim=0).to(self.device, dtype=self.dtype)  # (B, 3, 224, 224)
        questions = [self.txt_processor(question) for question in questions]

        query_input = {"image": images, "text_input": questions}
        query_embs = self.blip_model.extract_features(query_input, mode="multimodal")
        query_embs = query_embs.multimodal_embeds # (B, 32, 256)
        
        unique_entries, entry_embs = [], []
        for _entries in entries:
            _entries = self._remove_list_duplicates(_entries)
            unique_entries.append(_entries)

            if len(_entries) == 0:
                entry_embs.append(None)
            else:
                _entry_embs = []
                _entries = [self.txt_processor(entry) for entry in _entries]
                for idx in range(0, len(_entries), batch_size):
                    _entries_batch = _entries[idx:idx + batch_size]
                    _batch_embs = self.blip_model.extract_features({"text_input": _entries_batch}, mode="text")
                    _entry_embs.append(_batch_embs.text_embeds_proj[:, 0, :])
                
                _entry_embs = torch.concat(_entry_embs, dim=0) # (N, 256)
                entry_embs.append(_entry_embs)
        
        reranked_results = []
        for query_emb, entry_emb, _entries in zip(query_embs, entry_embs, unique_entries, strict=True):
            if entry_emb is None:
                reranked_results.append(([], []))
            else:
                scores = entry_emb @ query_emb.t() # (N, 32)
                scores, _ = scores.max(dim=-1) # pick the max score for each entry
                scores, rerank_indices = scores.sort(descending=True)
                scores = scores[scores > threshold]
                rerank_indices = rerank_indices[:len(scores)]
                reranked_entries = [_entries[i] for i in rerank_indices]
                reranked_results.append((reranked_entries, scores.tolist()))
        
        return reranked_results

    def rerank(self, 
        question: str, 
        image: Image.Image, 
        entries: List[str],
        threshold: float = 0,
    ) -> Tuple[List[str], List[float]]:
        """
        Rerank the retrieved entries based on their similarity scores.
        Args:
            question (str): The question asked.
            image (Image.Image): The image associated with the question.
            entries (list): List of retrieved entries.
        Returns:
            tuple: A tuple containing the reranked entries and their similarity scores.
        """
        if len(entries) == 0:
            return [], []

        entries = self._remove_list_duplicates(entries)

        image = self.vis_processor(image).unsqueeze(0).to(self.device, dtype=self.dtype)
        question = self.txt_processor(question)
        entries = [self.txt_processor(entry) for entry in entries]

        fusion_embs = self.blip_model.extract_features({"image": image, "text_input": question}, mode="multimodal")
        fusion_embs = fusion_embs.multimodal_embeds.squeeze(0) # (32, 256)

        entry_embs = []
        for entry in entries:
            entry_emb = self.blip_model.extract_features({"text_input": entry}, mode="text")
            entry_embs.append(entry_emb.text_embeds_proj[:, 0, :])
        
        entry_embs = torch.concat(entry_embs, dim=0) # (N, 256)
        scores = entry_embs @ fusion_embs.t() # (N, 32)

        scores, _ = scores.max(dim=-1) # pick the max score for each entry
        scores, rerank_indices = scores.sort(descending=True)
        scores = scores[scores > threshold]
        rerank_indices = rerank_indices[:len(scores)]

        reranked_entries = [entries[i] for i in rerank_indices]
        
        return reranked_entries, scores.tolist()


if __name__ == "__main__":
    # Example usage
    question = "What is the name of this store?"
    image = Image.open("./docs/dataset_info/multiqa.jpg")
    entries = ["This is a store that sells electronics.", 
               "Mount Fuji is an attractive volcanic cone. It has been a frequent subject of Japanese art, especially after 1600.",
               "Circle K Stores, Inc. is a Canadian-American chain of convenience stores headquartered in Tempe, Arizona."]
    
    reranker = Blip2Reranker()

    ## Test rerank
    reranked_entries, scores = reranker.rerank(question, image, entries)
    print("Results for non-batch reranking:")
    for idx, (entry, score) in enumerate(zip(reranked_entries, scores, strict=True)):
        print(f"Rank {idx + 1}: {entry}, Score: {score:.4f}")

    # Test rerank_batch
    reranked_results = reranker.rerank_batch([question] * 3, [image] * 3, [entries] * 2 + [[]])
    print("\nResults for batch reranking:")
    for result in reranked_results:
        reranked_entries, scores = result
        for idx, (entry, score) in enumerate(zip(reranked_entries, scores, strict=True)):
            print(f"Rank {idx + 1}: {entry}, Score: {score:.4f}")
