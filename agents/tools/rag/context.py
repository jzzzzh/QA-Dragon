import logging
from PIL import Image
from typing import List, Dict, Any, Tuple
from agents.tools.rag.chunking import TextSplitter
from agents.tools.rag.blip2_reranker import Blip2Reranker
from agents.tools.rag.qwen_reranker import QwenReranker


class ContextBuilder():
    def __init__(self, config):
        self.chunking_config = config.chunking.config
        self.reranker_config = config.reranker.config
        self.reranker_enabled = config.reranker.enabled

        self.chunk_size = self.chunking_config.get("chunk_size", 100)
        self.overlap_size = self.chunking_config.get("overlap_size", 30)
        self.text_splitter = TextSplitter(chunk_size=self.chunk_size, overlap_size=self.overlap_size)

        if self.reranker_enabled:
            logging.warning("Reranker is enabled in the context builder.")
            self.reranker = Blip2Reranker()
            if self.reranker_config.fine_rank:
                self.fine_reranker = QwenReranker()
        else:
            logging.warning("Reranker is disabled in the context builder.")
            self.reranker = None
            self.fine_reranker = None
    
    def _extract_chunks_from_image_results(self, results: List[Dict[str, Any]]) -> List[str]:
        chunks = []
        for result in results:
            entities = result.get("entities", [])
            for entity in entities:
                entity_name = entity.get("entity_name", None)
                entity_attributes = entity.get("entity_attributes", None)

                if entity_name is None or entity_attributes is None:
                    continue
                
                if isinstance(entity_attributes, str):
                    entity_attributes = {"description": entity_attributes}

                # chunk the long description
                description = entity_attributes.pop("description", None)
                if description:
                    chunks.extend([
                        chunk["content"] for chunk in self.text_splitter.article_chunking(description)
                    ])

                # chunk other attributes
                if entity_attributes:
                    chunks.extend([
                        f"The {chunk['title']} of {entity_name} is: {chunk['content']}"
                        for chunk in self.text_splitter.attribute_chunking(entity_attributes)
                    ])
        return chunks

    def _extract_chunks_from_text_results(self, results: List[Dict[str, Any]]) -> List[str]:
        chunks = []
        for result in results:
            ## {title}: {content} format
            # page_name = result.get("page_name", "")
            # page_snippet = result.get("page_snippet", "")
            # for chunk in self.text_splitter.article_chunking(page_snippet):
            #     title = chunk["title"] if chunk["title"] else page_name
            #     content = chunk["content"]
            #     chunks.append(f"{title}: {content}")

            ## only content
            page_snippet = result.get("page_snippet", "")
            chunks.extend([
                chunk["content"] for chunk in self.text_splitter.article_chunking(page_snippet)
            ])
        return chunks

    @staticmethod
    def _get_topk_results(
        results: List[Tuple[List[str], List[float]]],
        top_k: int,
    ) -> List[Tuple[List[str], List[float]]]:

        top_results = []
        for result in results:
            reranked_entries, scores = result
            reranked_entries = reranked_entries[: top_k]
            scores = scores[: top_k]
            top_results.append((reranked_entries, scores))
        
        return top_results

    def rerank(
        self,
        queries: List[str],
        images: List[Image.Image],
        chunks: List[List[str]]
    ) -> List[Tuple[List[str], List[float]]]:
        # coarse reranking
        results = self.reranker.rerank_batch(
            queries,
            images,
            chunks,
            threshold=self.reranker_config.coarse_min_score,
        )

        top_results = self._get_topk_results(results, self.reranker_config.coarse_top_k,)

        # fine reranking if enabled
        if self.reranker_config.fine_rank:
            entries, scores = zip(*top_results, strict=True)
            fine_results = self.fine_reranker.rerank_batch(
                queries,
                entries,
                weights=scores,
                threshold=self.reranker_config.fine_min_score,
            )
            top_results = self._get_topk_results(fine_results, self.reranker_config.fine_top_k)
        
        return top_results
    
    def naive_build(
        self,
        search_results: List[Dict[str, List[Dict[str, Any]]] | None],
    ):
        
        total_chunk_size = self.reranker_config.coarse_top_k * self.chunk_size
        batch_context = []
        for results in search_results:
            if results is None:
                batch_context.append(None)  # no context
                continue

            image_context, text_context = "", ""
            image_context_size, text_context_size = 0, 0
            for key, subresults in results.items():
                if key == "img_search":
                    chunks = self._extract_chunks_from_image_results(subresults)

                    if image_context_size >= total_chunk_size:
                        continue
                    
                    for chunk in chunks:
                        chunk_size = len(chunk.split())
                        if image_context_size + chunk_size > total_chunk_size:
                            image_context += f"\n {' '.join(chunk.split()[:total_chunk_size - image_context_size])}"
                            image_context_size = total_chunk_size
                            break
                        else:
                            image_context += f"\n {chunk}"
                            image_context_size += chunk_size

                elif key == "text_search":
                    chunks = self._extract_chunks_from_text_results(subresults)

                    if text_context_size >= total_chunk_size:
                        continue

                    for chunk in chunks:
                        chunk_size = len(chunk.split())
                        if text_context_size + chunk_size > total_chunk_size:
                            text_context += f"\n {' '.join(chunk.split()[:total_chunk_size - text_context_size])}"
                            text_context_size = total_chunk_size
                            break
                        else:
                            text_context += f"\n {chunk}"
                            text_context_size += chunk_size
                else:
                    raise ValueError(f"Unknown search result key: {key}")
            
            # combine image and text context
            combined_context = text_context.strip() + "\n\n" + image_context.strip()
            
            if len(combined_context.split()) > total_chunk_size:
                combined_context = " ".join(combined_context.split()[:total_chunk_size])

            batch_context.append(combined_context if combined_context else None)
        return batch_context
    
    def build(
        self, 
        queries: List[str],
        images: List[Image.Image],
        search_results: List[Dict[str, List[Dict[str, Any]]] | None],
    ) -> List[str]:
        
        batch_chunks = []
        for results in search_results:
            if results is None:
                batch_chunks.append([])  # no context
                continue

            chunks = []
            for key, subresults in results.items():
                if key == "img_search":
                    chunks.extend(self._extract_chunks_from_image_results(subresults))
                elif key == "text_search":
                    chunks.extend(self._extract_chunks_from_text_results(subresults))
                elif key == "history":
                    if isinstance(subresults, str):
                        subresults = [subresults]
                    chunks.extend(subresults)
                else:
                    raise ValueError(f"Unknown search result key: {key}")
            batch_chunks.append(chunks)

        top_results = self.rerank(queries, images, batch_chunks)
        
        # build context strings
        batch_context = []
        for result in top_results:
            reranked_entries, scores = result
            context = [
                f"Evidence {i+1} with relevance score {score:.2f}:\n{entry}"
                for i, (entry, score) in enumerate(zip(reranked_entries, scores, strict=True))
            ]
            
            if len(context) == 0:
                batch_context.append(None)
            else:
                batch_context.append("\n\n".join(context))
        
        return batch_context


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/hitech.yaml")
    context_builder = ContextBuilder(cfg.rag)

    queries = [
        "What is the capital of France?", 
        "Who is the president of the USA?",
        "What is wavelenth of red light?"
    ]
    
    images = [
        Image.open("./docs/dataset_info/multiqa.jpg"), 
        Image.open("./docs/dataset_info/multiqa.jpg"),
        Image.open("./docs/dataset_info/multiqa.jpg")
    ]

    search_results = [
        {
            "img_search": [
                {
                    "entities": [
                        {
                            "entity_name": "Eiffel Tower",
                            "entity_attributes": {
                                "description": "An iron lattice tower on the Champ de Mars in Paris, France.",
                                "height": "300 meters"
                            }
                        }
                    ]
                }
            ],
            "text_search": [
                {
                    "page_name": "France",
                    "page_snippet": "France is a country in Western Europe."
                },
                {
                    "page_name": "Paris",
                    "page_snippet": "Paris is the capital city of France."
                }
            ],
            "history": "The capital of France is Paris, known for the Eiffel Tower."
        },
        None,  # No results for the second query
        {
            "img_search": [
                {
                    "entities": [
                        {
                            "entity_name": "Red Light",
                            "entity_attributes": {
                                "description": "Red light has a wavelength of approximately 620-750 nm."
                            }
                        }
                    ]
                }
            ],
            "text_search": [
                {
                    "page_name": "Light Spectrum",
                    "page_snippet": "The visible light spectrum ranges from violet (380 nm) to red (750 nm)."
                }
            ]
        }
    ]
    
    context = context_builder.build(queries, images, search_results)

    for i, ctx in enumerate(context):
        print(f"Context for query {i+1}:\n{ctx}\n")
