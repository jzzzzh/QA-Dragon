from typing import List, Dict, Any
from PIL import Image
import sys
import logging
sys.path.append(".")
from cragmm_search.search import UnifiedSearchPipeline
from omegaconf import OmegaConf
from agents.tools.pre_processing import SingleImageSegmenter, MultiImageSegmenter
from agents.modules.filter import ImageSearchFilter
from agents.modules.text_search_generator import TextSearchGenerator, TextSearchEnhancer
from agents.modules.image_object_selection import PresetObjectSelectionAgent
from agents.utils.utils import extract_valid_samples
from agents.utils.image import resize_image
from agents.tools.rag import Blip2ITM
class Searcher:
    def __init__(self, search_engine, cfg, llm=None, tokenizer=None, formater=None, saver=None, image_segmenter_type: str = "simple", image_segmenter_kwargs: Dict[str, Any] = dict(type="simple", Edge_Threshold=1, Area_Threshold=0.02)):
        self.search_pipeline = search_engine
        self.topk = cfg.basic.num_search_result
        self.cfg = cfg
        self.llm = llm
        self.tokenizer = tokenizer
        self.formater = formater
        self.saver = saver
        self.image_segmenter = SingleImageSegmenter(config=self.cfg.preprocessing, **image_segmenter_kwargs) if image_segmenter_type == "simple" else MultiImageSegmenter(config=self.cfg.preprocessing, **image_segmenter_kwargs)
        self.text_search_generator = TextSearchGenerator(
            llm=self.llm,
            tokenizer=self.tokenizer,
            generation_config=self.cfg.generation_config.text_search_generator,
            formater=self.formater,
            num_queries=self.cfg.preprocessing.tools.text_search_generator.num_queries,
        )
        self.text_search_enhancer = TextSearchEnhancer(
            llm=self.llm,
            tokenizer=self.tokenizer,
            generation_config=self.cfg.generation_config.text_search_enhancer,
            formater=self.formater,
            num_queries=self.cfg.preprocessing.tools.text_search_enhancer.num_queries,
        )
        self.object_selection_agent = PresetObjectSelectionAgent(
            llm=self.llm,
            tokenizer=self.tokenizer,
            generation_config=self.cfg.generation_config.object_selection,
            formater=self.formater,
        )
        self.image_search_filter = ImageSearchFilter(
            llm=self.llm,
            tokenizer=self.tokenizer,
            generation_config=self.cfg.generation_config.image_search_filter,
            formater=self.formater,
        )
        self.blip2_itm = Blip2ITM()
        self.blip_threshold = 0.001
        logging.warning(f"using blip2_itm for image search filtering, the threshold is {self.blip_threshold}.")
        
    def _batch_img_search(
        self, images: List[Image.Image], result_all=None, label_name="IMG_search"
    ) -> List[List[Dict[str, Any]] | None]:
        if result_all is not None:
            results_all = result_all
        else:
            results_all: List[List[Dict[str, Any]] | None] = [
                {} for _ in range(len(images))
            ]
        if images == [] or images == None:
            return results_all 
        else:
            valid_idx, valid_imgs = extract_valid_samples(
                images, lambda x: (x is not None) and (x != [None])
            )
            img_rst = self.search_pipeline(
                list(valid_imgs), k=self.topk 
            )
            for k, idx in enumerate(list(valid_idx)):
                results_all[idx][label_name] = img_rst[k]
            return results_all

    def _batch_text_search(
        self, queries: List[str], result_all=None, label_name="query_search"
    ) -> List[List[Dict[str, Any]] | None]:
        if result_all is not None:
            results_all = result_all
        else:
            results_all: List[List[Dict[str, Any]] | None] = [
                {} for _ in range(len(queries))
            ]
        if queries == [] or queries == None:
            return results_all
        else:
            valid_idx, valid_queries = extract_valid_samples(
                queries, lambda x: x != "" and x != [None] and x != None
            )
            # print(f"valid_queries: {valid_queries}")
            
            query_rst = self.search_pipeline(
                list(valid_queries), k=self.topk 
            )
            for k, idx in enumerate(list(valid_idx)):
                results_all[idx][label_name] = query_rst[k]
            return results_all

    def covert_2d_list_to_1d_list(self, list_2d: List[List[Any]]):
        """
        Convert a 2D list to a 1D list and save the lengths of each sublist for reverse conversion.
        """
        list_1d = []
        lengths = []
        for sublist in list_2d:
            list_1d.extend(sublist)
            lengths.append(len(sublist))
        return list_1d, lengths

    def convert_1d_list_to_2d_list(self, list_1d: List[Any], lengths: List[int], type:str = "text", label_name: str = "search_result"):
        """
        Convert a 1D list back to a 2D list using the saved lengths.
        """
        list_2d = []
        start = 0
        for length in lengths:
            tmp_list = []
            end = start + length
            for i in range(start, end):
                # print(list_1d[i])
                if type == "text":
                    # print(f"list_1d[i]: {list_1d[i]}")
                    if list_1d[i].get(label_name, {}) == {}:
                        tmp_list.append({"page_name": "", "page_snippet": ""})
                    else:
                        for j in range(len(list_1d[i][label_name])):
                            tmp_list.append({"page_name" : list_1d[i][label_name][j]["page_name"], "page_snippet": list_1d[i][label_name][j]["page_snippet"]})
                elif type == "image":
                    # print(f"list_1d[i]: {list_1d[i]}")
                    if list_1d[i].get(label_name, {}) == {}:
                        tmp_list.append({"entities": []})
                    else:
                        for j in range(len(list_1d[i][label_name])):
                            tmp_list.append({"entities" : list_1d[i][label_name][j]["entities"]})
                elif type == "seg_answering":
                    tmp_list.append(list_1d[i])
            list_2d.append(tmp_list)
            start = end
        return list_2d

    def convert_simple_to_rag_format(self, list_1d: List[Any], type:str = "text", label_name: str = "search_result"):
        rag_input = []
        for rst_idx in list_1d:
            tmp_list = []
            if rst_idx == {} or rst_idx is None:
                # if type == "text":
                    # tmp_list.append({"page_name": "", "page_snippet": ""})
                # elif type == "image":
                    # tmp_list.append({"entities": []})
                rag_input.append(tmp_list)
                continue
            if label_name not in rst_idx:
                rag_input.append(tmp_list)
                continue
            if not rst_idx[label_name]:
                rag_input.append(tmp_list)
                continue
            for j in range(len(rst_idx[label_name])):
                if type == "text" and rst_idx[label_name][j]["score"] > self.cfg.basic.search_score_threshold:
                    if rst_idx[label_name][j]["page_name"] or rst_idx[label_name][j]["page_snippet"]:
                        tmp_list.append({"page_name" : rst_idx[label_name][j]["page_name"], "page_snippet": rst_idx[label_name][j]["page_snippet"]})
                elif type == "image" and rst_idx[label_name][j]["score"] > self.cfg.basic.search_score_threshold:
                    if rst_idx[label_name][j]["entities"]:
                        tmp_list.append({"entities" : rst_idx[label_name][j]["entities"]})
                # Remove duplicates by converting to a set of strings
                unique_items = {str(item) for item in tmp_list}
                # Convert back to list of dictionaries
                tmp_list = [eval(item) for item in unique_items]
            rag_input.append(tmp_list)
        return rag_input
                    
        
    def search(
        self,
        search_item_list: (
            List[Image.Image | str | List[Image.Image | str] | None] | None
        ) = None,
        label_name: str = "search_result",
        search_type: str = "image",
    ) -> List[List[Dict[str, Any]] | None]:
        """
        Perform a search operation based on the input type and format.
        This method supports searching for images, text, or nested lists of images or text.
        It delegates the search operation to appropriate helper methods based on the type
        of the input and converts the results into a format suitable for RAG (Retrieval-Augmented Generation).
        Args:
            search_item_list (List[Image.Image | str | List[Image.Image | str] | None] | None): 
                A list of search items, which can be:
                - A list of `Image.Image` objects for image search.
                - A list of strings for text search.
                - A nested list of `Image.Image` objects or strings for batch search.
                - None, indicating no search items.
            label_name (str): 
                A label name used for tagging the search results. Defaults to "search_result".
        Returns:
            List[List[Dict[str, Any]] | None]: 
                A list of search results in RAG format. The structure of the results depends on the input:
                - For a flat list of images or text, returns a flat list of dictionaries.
                - For a nested list of images or text, returns a nested list of dictionaries.
                - Returns an empty list if `search_item_list` is None.
        """
        # print(f"search_item_list: {search_item_list}, label_name: {label_name}")
        # if search_item_list is None:
        #     return []
        # else:
        if any(isinstance(item, Image.Image) for item in search_item_list):
            search = self._batch_img_search(search_item_list, label_name=label_name)
            return self.convert_simple_to_rag_format(
                search, type="image", label_name=label_name
            )
        elif any(isinstance(item, str) for item in search_item_list):
            search = self._batch_text_search(search_item_list, label_name=label_name)
            return self.convert_simple_to_rag_format(
                search, type="text", label_name=label_name
            )
        elif any(isinstance(item, list) for item in search_item_list):
            if any(isinstance(sub_item, Image.Image) for sub_item in search_item_list[0]):
                d1_search_item_list, lengths = self.covert_2d_list_to_1d_list(
                    search_item_list
                )
                d1_search_result = self._batch_img_search(
                    d1_search_item_list, label_name=label_name
                )
                d2_search_result = self.convert_1d_list_to_2d_list(
                    d1_search_result, lengths, type="image", label_name=label_name
                )
                return d2_search_result
            elif any(isinstance(sub_item, str) for sub_item in search_item_list[0]):
                d1_search_item_list, lengths = self.covert_2d_list_to_1d_list(
                    search_item_list
                )
                d1_search_result = self._batch_text_search(
                    d1_search_item_list, label_name=label_name
                )
                d2_search_result = self.convert_1d_list_to_2d_list(
                    d1_search_result, lengths, type="text", label_name=label_name
                )
                return d2_search_result
            else:
                search = [{} for _ in range(len(search_item_list))]
                return self.convert_simple_to_rag_format(
                    search, type=search_type, label_name=label_name
                )
        else:
            search = [{} for _ in range(len(search_item_list))]
            return self.convert_simple_to_rag_format(
                search, type=search_type, label_name=label_name
            )
            
    def img_search_agent(self, images, queries, image_search_indices, origin_need_rag_images=None):
        assert len(images) == len(queries) and (not origin_need_rag_images or len(origin_need_rag_images) == len(images) == len(queries)), f"{len(origin_need_rag_images) =}, {len(images) =}, {len(queries) =}, {len(image_search_indices) =}"
        image_search_images = [images[idx] for idx in image_search_indices]
        image_search_queries = [queries[idx] for idx in image_search_indices]
        origin_need_rag_images = [origin_need_rag_images[idx] for idx in image_search_indices] if origin_need_rag_images else None
        
        assert len(image_search_images) == len(image_search_queries) == len(image_search_indices) and (not origin_need_rag_images or len(origin_need_rag_images) == len(image_search_images)), f"{len(image_search_images) =}, {len(image_search_queries) =}, {len(image_search_indices) =}, {len(origin_need_rag_images) =}"
        
        image_search_results = [None] * len(images)
        selected_objects, raw_selected_objects = self.object_selection_agent(
            images=image_search_images,
            queries=image_search_queries
        )
        # TODO: checker?
        self.saver.save_str_list(selected_objects, prefix="selected_objects", indices=image_search_indices)
        self.saver.save_str_list(raw_selected_objects, prefix="selected_objects_raw", indices=image_search_indices)
        segmented_images = self.image_segmenter.batch_segment_images(source_list=image_search_images, names=selected_objects, origin_source_list=origin_need_rag_images)
        self.saver.save_image_list(segmented_images, prefix="tmp_segmented_image", indices=image_search_indices)
        temp_image_search_results = self.search(segmented_images, search_type="image")
        # print(f"temp_image_search_results: {temp_image_search_results}, len: {len(temp_image_search_results)}")
        self.saver.save_str_list(temp_image_search_results, prefix="image_search_results_raw", indices=image_search_indices)
        image_object_name, filtered_temp_image_search_results, raw_filtered_temp_image_search_results = self.image_search_filter(image_search_images, temp_image_search_results, segment_categories=selected_objects)
        self.saver.save_str_list(image_object_name, prefix="filtered_image_object_name", indices=image_search_indices)
        self.saver.save_str_list(filtered_temp_image_search_results, prefix="image_search_results_filtered", indices=image_search_indices)
        self.saver.save_str_list(raw_filtered_temp_image_search_results, prefix="image_search_results_filtered_raw", indices=image_search_indices)
        self.saver.save_image_list(segmented_images, prefix="segmented_image", indices=image_search_indices)
        assert len(filtered_temp_image_search_results) == len(image_search_indices)
        for idx, filtered_temp_image_search_result in zip(image_search_indices, filtered_temp_image_search_results, strict=True):
            image_search_results[idx] = filtered_temp_image_search_result
        return image_search_results, image_object_name
    
    @property
    def black_list(self):
        return ["this ", " this?", "that ", " that?", "it ", " it?", "these ", " these?", "those ", " those?", " in image", "in image", " in the image", "in the image"]
    
    def filter_sub_questions(self, queries:List[str], enhanced_text_search_generated_queries:List[List[str]]) -> List[List[str]]:
        """
        Filter out sub-questions that are not relevant to the original query.
        """
        filtered_queries = []
        for original_query, sub_queries in zip(queries, enhanced_text_search_generated_queries, strict=True):
            filtered_sub_queries = []
            sub_lens = len(sub_queries)
            for idx, sub_query in enumerate(sub_queries):
                if any(word in sub_query.lower() for word in self.black_list) and sub_lens > 1:
                    # If the sub-query contains pronouns, skip it unless it's the only sub-query
                    if len(filtered_sub_queries) == 0:
                        filtered_sub_queries.append(sub_query)
                    continue
                if len(set(original_query.lower().split()) & set(sub_query.lower().split())) / len(set(original_query.lower().split())) >= self.cfg.searcher.config.similarity_threshold and sub_lens > 1:
                    # If the sub-query is too similar to the original query, skip it
                    if len(filtered_sub_queries) == 0:
                        filtered_sub_queries.append(sub_query)
                    continue
                filtered_sub_queries.append(sub_query)
            filtered_queries.append(filtered_sub_queries)
        return filtered_queries
        
    
    def concat_keyword_search_queries(self, sub_queries_list:List[List[str]], image_object_name_list:List[str|None]=None, text_search_indices= None, image_search_indices=None) -> List[List[str]]:
        concatenated_queries = []
        if image_object_name_list is None:
            return sub_queries_list
        else:
            assert len(sub_queries_list) == len(text_search_indices), f"{len(sub_queries_list) =}, {len(text_search_indices) =}"
            assert len(image_search_indices) == len(image_object_name_list), f"{len(image_search_indices) =}, {len(image_object_name_list) =}"
            for idx, sub_queries in zip(text_search_indices, sub_queries_list, strict=True):
                tmp_sub_queries = sub_queries
                if idx in image_search_indices:
                    tmp_sub_queries.append(image_object_name_list[image_search_indices.index(idx)])
                concatenated_queries.append(tmp_sub_queries)
        return concatenated_queries
                    
    
    def text_search_agent(self, images, queries, text_search_indices, image_rag_results = None, image_object_names=None, image_search_indices=None):
        text_search_images = [images[idx] for idx in text_search_indices]
        text_search_original_queries = [queries[idx] for idx in text_search_indices]
        assert len(text_search_images) == len(text_search_original_queries) == len(text_search_indices) and (not image_rag_results or len(image_rag_results) == len(text_search_images)), f"{len(text_search_images) =}, {len(text_search_original_queries) =}, {len(text_search_indices) =}, {len(image_rag_results) =}"
        text_search_results = [None] * len(images)
        text_search_generated_queries, raw_text_search_generated_queries = self.text_search_generator(images=text_search_images, queries=text_search_original_queries, image_rag_results=image_rag_results)
        enhanced_text_search_generated_queries, raw_enhanced_text_search_generated_queries = self.text_search_enhancer(
            images=text_search_images, queries=text_search_original_queries, subqueries=text_search_generated_queries
        )
        filtered_queries = self.filter_sub_questions(text_search_original_queries, enhanced_text_search_generated_queries)
        if image_object_names and image_search_indices:
            filtered_queries = self.concat_keyword_search_queries(filtered_queries, image_object_names, text_search_indices=text_search_indices, image_search_indices=image_search_indices)
        self.saver.save_str_list(list(map(str, text_search_generated_queries)), prefix="text_search_queries", indices=text_search_indices)
        self.saver.save_str_list(raw_text_search_generated_queries, prefix="text_search_queries_raw", indices=text_search_indices)
        self.saver.save_str_list(list(map(str, enhanced_text_search_generated_queries)), prefix="text_search_queries_enhanced", indices=text_search_indices)
        self.saver.save_str_list(raw_enhanced_text_search_generated_queries, prefix="text_search_queries_enhanced_raw", indices=text_search_indices)
        self.saver.save_str_list(list(map(str, filtered_queries)), prefix="text_search_queries_filtered", indices=text_search_indices)
        temp_text_search_results = self.search(filtered_queries, search_type="text")
        self.saver.save_str_list(temp_text_search_results, prefix="text_search_results", indices=text_search_indices)
        assert len(temp_text_search_results) == len(text_search_indices)
        for idx, result in zip(text_search_indices, temp_text_search_results, strict=True):
            text_search_results[idx] = result
        assert len(text_search_results) == len(images)
        return text_search_results
    



class Searcher_v2(Searcher):
    """
    A more advanced version of the Searcher class that supports both image and text search.
    It can handle nested lists of images or text, and performs searches accordingly.
    """
    def __init__(self, search_engine, cfg, llm=None, tokenizer=None, formater=None, saver=None, answer_agent = None,  need_seg_preans = False, image_segmenter_type: str = "simple", image_segmenter_kwargs: Dict[str, Any] = dict(type="simple", Edge_Threshold=1, Area_Threshold=0.02)):
        super().__init__(search_engine, cfg, llm, tokenizer, formater, saver, image_segmenter_type, image_segmenter_kwargs)
        self.search_version = "v2"
        self.answer_agent = answer_agent
        self.need_seg_preans = need_seg_preans
        self.visit = {}
    
    def filter_sub_questions(self, queries:List[str], enhanced_text_search_generated_queries:List[List[str]],  mode = "soft"):
        """
        Filter out sub-questions that are not relevant to the original query.
        """
        filtered_queries = []
        need_regenerate_with_img_rag_indices = []
        for idx, (original_query, sub_queries) in enumerate(zip(queries, enhanced_text_search_generated_queries, strict=True)):
            filtered_sub_queries = []
            sub_lens = len(sub_queries)
            for sub_query in sub_queries:
                if any(word in sub_query.lower() for word in self.black_list) and sub_lens > 1:
                    if mode == "soft" and len(filtered_sub_queries) == 0:
                        filtered_sub_queries.append(sub_query)
                    continue
                if len(set(original_query.lower().split()) & set(sub_query.lower().split())) / len(set(original_query.lower().split())) >= self.cfg.searcher.config.similarity_threshold and sub_lens > 1:
                    if mode == "soft" and len(filtered_sub_queries) == 0:
                        filtered_sub_queries.append(sub_query)
                    continue
                filtered_sub_queries.append(sub_query)
            if len(filtered_sub_queries) == 0:
                need_regenerate_with_img_rag_indices.append(idx)
            filtered_queries.append(filtered_sub_queries)
        return filtered_queries, need_regenerate_with_img_rag_indices
    
    # def regenerate_with_img_rag(self, text_search_images, text_search_original_queries, need_regenerate_with_img_rag_indices, image_rag_results, filtered_queries, text_search_reasoning):
    #     # This method handles the regeneration of queries using image-based RAG (Retrieval-Augmented Generation) results.
    #     # It processes the subset of queries that require regeneration due to insufficient or irrelevant sub-queries.
    #     # The regenerated queries are enhanced and filtered before being integrated back into the original query list.
    #     # First gen subquery If the query doesn't have a clear word, then add img result
    #     need_regenerate_with_img_rag_images = [text_search_images[idx] for idx in need_regenerate_with_img_rag_indices]
    #     need_regenerate_with_img_rag_queries = [text_search_original_queries[idx] for idx in need_regenerate_with_img_rag_indices]
    #     need_regenerate_with_img_rag_reasoning = [text_search_reasoning[idx] for idx in need_regenerate_with_img_rag_indices] if text_search_reasoning else None
    #     need_regenerate_with_img_rag_image_rag_results = [image_rag_results[idx] for idx in need_regenerate_with_img_rag_indices] if image_rag_results else None
    #     assert len(need_regenerate_with_img_rag_images) == len(need_regenerate_with_img_rag_queries) == len(need_regenerate_with_img_rag_indices) and (not need_regenerate_with_img_rag_image_rag_results or len(need_regenerate_with_img_rag_image_rag_results) == len(need_regenerate_with_img_rag_images)), f"{len(need_regenerate_with_img_rag_images) =}, {len(need_regenerate_with_img_rag_queries) =}, {len(need_regenerate_with_img_rag_indices) =}, {len(need_regenerate_with_img_rag_image_rag_results) =}"
    #     if len(need_regenerate_with_img_rag_images) == 0:
    #         return filtered_queries
    #     regenerated_queries, raw_regenerated_queries = self.text_search_generator(
    #         images=need_regenerate_with_img_rag_images, queries=need_regenerate_with_img_rag_queries, reasonings=need_regenerate_with_img_rag_reasoning, image_rag_results=need_regenerate_with_img_rag_image_rag_results
    #     )
    #     enhanced_regenerated_queries, raw_enhanced_regenerated_queries = self.text_search_enhancer(
    #         images=need_regenerate_with_img_rag_images, queries=need_regenerate_with_img_rag_queries, subqueries=regenerated_queries
    #     )
    #     self.saver.save_str_list(list(map(str, regenerated_queries)), prefix="regenerated_text_search_queries", indices=need_regenerate_with_img_rag_indices)
    #     self.saver.save_str_list(raw_regenerated_queries, prefix="regenerated_text_search_queries_raw", indices=need_regenerate_with_img_rag_indices)
    #     self.saver.save_str_list(list(map(str, enhanced_regenerated_queries)), prefix="regenerated_text_search_queries_enhanced", indices=need_regenerate_with_img_rag_indices)
    #     self.saver.save_str_list(raw_enhanced_regenerated_queries, prefix="regenerated_text_search_queries_enhanced_raw", indices=need_regenerate_with_img_rag_indices)
    #     filtered_regenerated_queries, _ = self.filter_sub_questions(need_regenerate_with_img_rag_queries, enhanced_regenerated_queries, mode="soft")
    #     self.saver.save_str_list(list(map(str, filtered_regenerated_queries)), prefix="filtered_regenerated_text_search_queries", indices=need_regenerate_with_img_rag_indices)
    #     for idx, regenerated_query in zip(need_regenerate_with_img_rag_indices, filtered_regenerated_queries ,strict=True):
    #         filtered_queries[idx] = regenerated_query
    #     return filtered_queries
    
    
    def img_search_agent(self, images, queries, image_search_indices, origin_need_rag_images=None):
        assert len(images) == len(queries) and (not origin_need_rag_images or len(origin_need_rag_images) == len(images) == len(queries)), f"{len(origin_need_rag_images) =}, {len(images) =}, {len(queries) =}, {len(image_search_indices) =}"
        image_search_images = [images[idx] for idx in image_search_indices]
        image_search_queries = [queries[idx] for idx in image_search_indices]
        seg_img_answersing = [""] * len(images)
        origin_need_rag_images = [origin_need_rag_images[idx] for idx in image_search_indices] if origin_need_rag_images else None
        
        assert len(image_search_images) == len(image_search_queries) == len(image_search_indices) and (not origin_need_rag_images or len(origin_need_rag_images) == len(image_search_images)), f"{len(image_search_images) =}, {len(image_search_queries) =}, {len(image_search_indices) =}, {len(origin_need_rag_images) =}"
        
        image_search_results = [""] * len(images)
        selected_objects, raw_selected_objects = self.object_selection_agent(
            images=image_search_images,
            queries=image_search_queries
        )
        self.saver.save_str_list(selected_objects, prefix="selected_objects", indices=image_search_indices)
        self.saver.save_str_list(raw_selected_objects, prefix="selected_objects_raw", indices=image_search_indices)
        segmented_images = self.image_segmenter.batch_segment_images(source_list=image_search_images, names=selected_objects, origin_source_list=origin_need_rag_images)
        # self.saver.save_image_list(segmented_images, prefix="tmp_segmented_image", indices=image_search_indices)
        assert len(segmented_images) == len(image_search_indices), f"{len(segmented_images) =}, {len(image_search_indices) =}"
        seg_img_answersing_indices = [""] * len(image_search_queries)
        # NOTE: Segmented images = List[Image.Image] or List[List[Image.Image]]
        # NOTE: Close the segmented_images pre-answering.
        if self.need_seg_preans:
            if isinstance(segmented_images[0], list):
                segmented_images_1d, lengths = self.covert_2d_list_to_1d_list(segmented_images)
                assert len(segmented_images_1d) == sum(lengths), f"{len(segmented_images_1d) =}, {sum(lengths) =}"
                assert len(image_search_queries) == len(lengths), f"{len(image_search_queries) =}, {len(lengths) =}"
                image_search_queries_1d = [image_search_queries[idx] for idx in range(len(lengths)) for _ in range(lengths[idx])]
                assert len(image_search_queries_1d) == sum(lengths), f"{len(image_search_queries) =}, {sum(lengths) =}"
            else:
                segmented_images_1d = segmented_images
                image_search_queries_1d = image_search_queries
                lengths = None
            segmented_images_1d = [resize_image(img, self.cfg.basic.max_img_size) for img in segmented_images_1d]
            if segmented_images_1d:
                seg_img_answersing_indices, raw_responses = self.answer_agent(
                    segmented_images_1d, 
                    image_search_queries_1d,
                )
                if lengths:
                    seg_img_answersing_indices = self.convert_1d_list_to_2d_list(seg_img_answersing_indices, lengths, type="seg_answering", label_name="segmented_image_answering")
                    raw_responses = self.convert_1d_list_to_2d_list(raw_responses, lengths, type="seg_answering", label_name="segmented_image_answering")
                self.saver.save_str_list(seg_img_answersing_indices, prefix="segmented_image_answering", indices=image_search_indices)
                self.saver.save_str_list(raw_responses, prefix="segmented_image_answering_raw", indices=image_search_indices)
        segmented_images = self.blip2_itm(image_search_queries, segmented_images, Threshold=self.blip_threshold)
        temp_image_search_results = self.search(segmented_images, search_type="image")
        # print(f"temp_image_search_results: {temp_image_search_results}, len: {len(temp_image_search_results)}")
        self.saver.save_str_list(temp_image_search_results, prefix="image_search_results_raw", indices=image_search_indices)
        if any(isinstance(inner, list) for item in temp_image_search_results for inner in item):
            # Multiple objects segmented, flatten the nested list
            temp_image_search_results = [
            [inner_inner_item for inner_item in item for inner_inner_item in inner_item]
            for item in temp_image_search_results
            ]
        image_object_name, filtered_temp_image_search_results, raw_filtered_temp_image_search_results = self.image_search_filter(image_search_images, temp_image_search_results, segment_categories=selected_objects)
        self.saver.save_str_list(image_object_name, prefix="filtered_image_object_name", indices=image_search_indices)
        self.saver.save_str_list(filtered_temp_image_search_results, prefix="image_search_results_filtered", indices=image_search_indices)
        self.saver.save_str_list(raw_filtered_temp_image_search_results, prefix="image_search_results_filtered_raw", indices=image_search_indices)
        self.saver.save_image_list(segmented_images, prefix="segmented_image", indices=image_search_indices)
        assert len(filtered_temp_image_search_results) == len(image_search_indices)
        for idx, filtered_temp_image_search_result, seg_ans in zip(image_search_indices, filtered_temp_image_search_results, seg_img_answersing_indices, strict=True):
            image_search_results[idx] = filtered_temp_image_search_result
            seg_img_answersing[idx] = seg_ans
            assert len(image_search_results) == len(images) and len(seg_img_answersing) == len(images), f"{len(image_search_results) =}, {len(images) =}, {len(seg_img_answersing) =}"
        return image_search_results, image_object_name, seg_img_answersing
    
    
    def text_search_agent(self, images, queries, text_search_indices, image_rag_results=None, reasoning=None, image_object_names=None, image_search_indices=None):
        """
        Perform text search and query generation on the provided images and queries.
        Note: segmented_images is not equal to batch size. equal to text_search_indices size.
        """
        text_search_images = [images[idx] for idx in text_search_indices]
        text_search_original_queries = [queries[idx] for idx in text_search_indices]
        text_search_reasoning = [reasoning[idx] for idx in text_search_indices]
        text_search_image_rag_results = [image_rag_results[idx] for idx in text_search_indices] if image_rag_results else None
        assert len(text_search_images) == len(text_search_original_queries) == len(text_search_indices) and (not text_search_image_rag_results or len(text_search_image_rag_results) == len(text_search_images)), f"{len(text_search_images) =}, {len(text_search_original_queries) =}, {len(text_search_indices) =}, {len(text_search_image_rag_results) =}"
        text_search_results = [None] * len(images)
        text_search_generated_queries, raw_text_search_generated_queries = self.text_search_generator(images=text_search_images, queries=text_search_original_queries, reasonings=text_search_reasoning, image_rag_results=text_search_image_rag_results)
        enhanced_text_search_generated_queries, raw_enhanced_text_search_generated_queries = self.text_search_enhancer(
            images=text_search_images, queries=text_search_original_queries, subqueries=text_search_generated_queries
        )
        filtered_queries, need_regenerate_indices = self.filter_sub_questions(text_search_original_queries, enhanced_text_search_generated_queries, mode="hard")
        # filtered_queries = self.regenerate_with_img_rag(images, queries, need_regenerate_indices, image_rag_results, filtered_queries, reasoning)
        if image_object_names and image_search_indices:
            filtered_queries = self.concat_keyword_search_queries(filtered_queries, image_object_names, text_search_indices=text_search_indices, image_search_indices=image_search_indices)
        self.saver.save_str_list(list(map(str, text_search_generated_queries)), prefix="text_search_queries", indices=text_search_indices)
        self.saver.save_str_list(raw_text_search_generated_queries, prefix="text_search_queries_raw", indices=text_search_indices)
        self.saver.save_str_list(list(map(str, enhanced_text_search_generated_queries)), prefix="text_search_queries_enhanced", indices=text_search_indices)
        self.saver.save_str_list(raw_enhanced_text_search_generated_queries, prefix="text_search_queries_enhanced_raw", indices=text_search_indices)
        self.saver.save_str_list(list(map(str, filtered_queries)), prefix="text_search_queries_filtered", indices=text_search_indices)
        self.visit["filtered_queries"] = filtered_queries
        temp_text_search_results = self.search(filtered_queries, search_type="text")
        self.saver.save_str_list(temp_text_search_results, prefix="text_search_results", indices=text_search_indices)
        assert len(temp_text_search_results) == len(text_search_indices)
        for idx, result in zip(text_search_indices, temp_text_search_results, strict=True):
            text_search_results[idx] = result
        assert len(text_search_results) == len(images), f"{len(text_search_results) =}, {len(images) =}"
        return text_search_results
    
    def get_visit(self):
        """
        Get the visit dictionary.
        """
        return self.visit
    

if __name__ == "__main__":
    search_api_text_model = "BAAI/bge-large-en-v1.5"
    search_api_image_model = "openai/clip-vit-large-patch14-336"
    web_index = "crag-mm-2025/web-search-index-validation"
    image_index = "crag-mm-2025/image-search-index-validation"
    config_file = "configs/hitech.yaml"
    cfg = OmegaConf.load(config_file)
    # ------ Build pipeline & agent
    search_pipeline = UnifiedSearchPipeline(
        text_model_name=search_api_text_model,
        image_model_name=search_api_image_model,
        web_hf_dataset_id=web_index,
        image_hf_dataset_id=image_index,
    )
    searcher = Searcher(search_pipeline, cfg)
    # ------ Test
    test_images = [
        Image.open("docs/dataset_info/singleqa.jpg"),  Image.open("docs/dataset_info/singleqa.jpg"), None
    ]   # Replace with actual image paths
    test_queries = ["What is this?", "Describe the image.", None, "What can you see?"]
    test_none = [None, None, None, None]
    results = searcher.search(search_item_list=test_queries)
    print("================TEST1=======================")
    for re in results:
        print(len(re))
        print(re)
        print("=======================================")
    results = searcher.search(search_item_list=test_images)
    print("================TEST2=======================")
    for re in results:
        print(len(re))
        print(re)
        print("=======================================")
    # TODO: add None???
    test_images = [
        [Image.open("docs/dataset_info/singleqa.jpg")], [Image.open("docs/dataset_info/singleqa.jpg"), Image.open("docs/dataset_info/singleqa.jpg")]
    ]
    results = searcher.search(search_item_list=test_images)
    print("================TEST3=======================")
    for re in results:
        print(len(re))
        print(re)
        print("=======================================")
    test_queries = [["What is this?", "Describe the image."], ["What can you see?"]]
    results = searcher.search(search_item_list=test_queries)
    print("================TEST4=======================")
    # print(results)
    for re in results:
        print(len(re))
        print(re)
        print("=======================================")
    results = searcher.search(search_item_list=test_none)
    print("================TEST5=======================")
    print(results)
    
    
    rst = [
        {
            "search_result": [
                {"page_name": "Page 1", "page_snippet": "Snippet 1", "score": 0.9},
                {"page_name": "Page 2", "page_snippet": "Snippet 2", "score": 0.8}
            ]
        },
        {
            "search_result": [
                {"page_name": "Page 3", "page_snippet": "Snippet 3", "score": 0.7},
                {"page_name": "", "page_snippet": "", "score": 0.0}
            ]
        }
    ]
    print("case 1:")
    rst = searcher.convert_simple_to_rag_format(rst, type="text")
    print(rst)
    print(len(rst))
    
    rst = [
        {
            "search_result": [
                {"entities": ["Entity 1", "Entity 2"], "score": 0.9},
                {"entities": [], "score": 0.0}
            ]
        },
        {
            "search_result": [
                {"entities": ["Entity 3"], "score": 0.8},
                {"entities": ["Entity 4"], "score": 0.7}
            ]
        }
    ]
    print("case 2:")
    rst = searcher.convert_simple_to_rag_format(rst, type="image")
    print(rst)
    print(len(rst))
    rst = [
        {
            "search_result": []
        },
        {
            "search_result": None
        }
    ]
    print("case 3:")
    rst = searcher.convert_simple_to_rag_format(rst, type="text")
    print(rst)
    print(len(rst))
    rst = [
        {
            "search_result": [
                {"page_name": "Page 1", "page_snippet": "Snippet 1", "score": 0.9},
                {"page_name": "Page 1", "page_snippet": "Snippet 1", "score": 0.9}
            ]
        }
    ]
    print("case 4:")
    rst = searcher.convert_simple_to_rag_format(rst, type="text")
    print(rst)
    print(len(rst))
    rst = [
        {
            "search_result": [
                {"entities": ["Entity 1", "Entity 2"], "score": 0.9},
                {"entities": ["Entity 1", "Entity 2"], "score": 0.9}
            ]
        }
    ]
    print("case 5:")
    rst = searcher.convert_simple_to_rag_format(rst, type="image")
    print(rst)
    print(len(rst))
    rst = [
        {
            "search_result": [
                {"page_name": "Page 1", "page_snippet": "Snippet 1", "score": 0.9},
                {"page_name": "", "page_snippet": "", "score": 0.0}
            ]
        }
    ]
    print("case 6:")
    rst = searcher.convert_simple_to_rag_format(rst, type="text")
    print(rst)
    print(len(rst))