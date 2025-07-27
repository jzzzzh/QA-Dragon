import logging
import os
import sys
import time
from typing import Any, Dict, List

import vllm
from cragmm_search.search import UnifiedSearchPipeline
from omegaconf import OmegaConf
from PIL import Image
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

sys.path.append(".")
from agents.base import MetaBaseAgent
from agents.modules.answer import DomainAwareReasoningAgent
from agents.modules.image_object_selection import PresetObjectSelectionAgent
from agents.modules.routers.answerable import RuleAnswerableRouter, ReasoningAnswerableChecker
from agents.modules.routers.tool import ImageTextSearchToolRouter
from agents.modules.text_search_generator import TextSearchGenerator, VerifyTextSearchGenerator
from agents.modules.filter import ImageSearchFilter, QueryFilter
from agents.modules.routers.blip2_router import BLIP2Router
from agents.tools.post_processing.verifier import Verifier
from agents.tools.saver.saver_v2 import Saver_v2
from agents.tools.rag.context import ContextBuilder
from agents.tools.rag.generator import AnswerGenerator
from agents.tools.search import Searcher_v2
from agents.utils.formater import Formater
from agents.utils.image import resize_image, repair_images


class QADragon4Multi(MetaBaseAgent):

    def __init__(self, search_pipeline=None, domain_aware: bool = False, all_text_verify: bool = True, enable_saver: bool = False, logging_level: str = "WARNING", use_black_list: bool = False):
        
        super().__init__(search_pipeline)
        self.use_text_search = True
        cfg_file = "configs/qadragon.yaml"
        cfg = OmegaConf.load(cfg_file)
        self.cfg = cfg
        self.cfg.saver.enabled = enable_saver
        agent_name = 'qadragon4_agent'
        self.saver = Saver_v2(agent_name=agent_name, cfg=self.cfg, logging_level=logging_level)
        self.model_name = self.cfg.basic.model_name
        self.domain_aware = domain_aware
        self.all_text_verify = all_text_verify
        self.use_black_list = use_black_list
        self.register = dict()
        self.direct_black_list_symbols = ['{']
        self._init_models()
        
        
    def _init_models(self):
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=self.cfg.basic.VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=self.cfg.basic.VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len=self.cfg.basic.MAX_MODEL_LEN,
            max_num_seqs=self.cfg.basic.MAX_NUM_SEQS,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1},
            task="generate",
        )
        self.tokenizer = self.llm.get_tokenizer()
        # ================ Tools ================
        self.rag_context_builder = ContextBuilder(config=self.cfg.rag)
        self.answer_agent = AnswerGenerator(llm=self.llm, tokenizer=self.tokenizer, config=self.cfg.rag.generation)
        
        self.verifier = Verifier(llm=self.llm, tokenizer=self.tokenizer, config=self.cfg.postprocessing.verifier)
        # ================ Modules ================
        self.formater = Formater(
            llm=self.llm,
            tokenizer=self.tokenizer,
            generation_config=self.cfg.generation_config.formater,
        )
        if self.domain_aware:
            self.domain_router = BLIP2Router()
        self.pre_answer_agent = DomainAwareReasoningAgent(
            llm=self.llm,
            tokenizer=self.tokenizer,
            generation_config=self.cfg.generation_config.answer_agent,
            formater=self.formater,
        )
        self.answerable_agent = RuleAnswerableRouter(
            all_text_verify=self.all_text_verify,
        )
        self.tool_router = ImageTextSearchToolRouter(
            llm=self.llm,
            tokenizer=self.tokenizer,
            generation_config=self.cfg.generation_config.tool_router,
            formater=self.formater,
        )
        self.image_search_filter = ImageSearchFilter(
            llm=self.llm,
            tokenizer=self.tokenizer,
            generation_config=self.cfg.generation_config.image_search_filter,
            formater=self.formater,
        )
        self.text_search_generator = TextSearchGenerator(
            llm=self.llm,
            tokenizer=self.tokenizer,
            generation_config=self.cfg.generation_config.text_search_generator,
            formater=self.formater,
            num_queries=self.cfg.preprocessing.tools.text_search_generator.num_queries,
        )
        self.object_selection_agent = PresetObjectSelectionAgent(
            llm=self.llm,
            tokenizer=self.tokenizer,
            generation_config=self.cfg.generation_config.object_selection,
            formater=self.formater,
        )
        self.searcher = Searcher_v2(
            search_engine=self.search_pipeline, 
            cfg=self.cfg, 
            llm=self.llm, 
            tokenizer=self.tokenizer, 
            formater=self.formater, 
            saver=self.saver, 
            answer_agent=self.pre_answer_agent,
            image_segmenter_type="simple",
            image_segmenter_kwargs={"type":"hard", "Edge_Threshold": 1, "Area_Threshold": 0.02}
        )
        self.verify_text_search_generator = VerifyTextSearchGenerator(
            llm=self.llm,
            tokenizer=self.tokenizer,
            generation_config=self.cfg.generation_config.text_search_generator,
            formater=self.formater,
            num_queries=self.cfg.preprocessing.tools.text_search_generator.num_queries
        )
        self.query_filter = QueryFilter()
        self.reasoning_answerable_checker = ReasoningAnswerableChecker()
    def get_batch_size(self) -> int:
        return self.cfg.basic.batch_size
    
    
    def concat_search_queries(self, subqueries: List[List[str]], queries: List[str], indices: List[int]) -> List[str]:
        """
        Concatenate subqueries with the main queries.
        """
        if not subqueries:
            return queries
        assert len(subqueries) == len(indices), "Subqueries length must match indices length."
        concatenated_queries = queries.copy()
        for idx, subquery_list in zip(indices, subqueries, strict=True):
            if not subquery_list:
                concatenated_queries[idx] = queries[idx]
            else:
                subquery_list = [query for query in subquery_list if isinstance(query, str) and query.strip()]
                concatenated_query = f"{concatenated_queries[idx]} {'? '.join(subquery_list)}"
                concatenated_queries[idx] = concatenated_query
        return concatenated_queries
    
    def rag_process(self, images, queries, reasonings, indices, origin_images):
        final_answers = ['I don\'t know' for _ in range(len(queries))]
        rag_images = [images[idx] for idx in indices]   
        rag_queries = [queries[idx] for idx in indices]
        rag_reasonings = [reasonings[idx] for idx in indices]
        decisions, raw_decisions = self.tool_router(images=rag_images, queries=rag_queries, reasonings=rag_reasonings)
        self.saver.save_str_list(list(map(str, decisions)), prefix="tool_router_decisions", indices=indices)
        self.saver.save_str_list(raw_decisions, prefix="tool_router_decisions_raw", indices=indices)
        
        image_search_indices = [idx for idx, decision in zip(indices, decisions, strict=True) if decision[0]]
        if self.use_text_search:
            text_search_indices = [idx for idx, decision in zip(indices, decisions, strict=True) if decision[1]]
        else:
            text_search_indices = []
        # search for images
        if image_search_indices:
            image_search_results, image_object_name, _ = self.searcher.img_search_agent(images, queries, image_search_indices, origin_images)
        else:
            image_object_name = [None] * len(queries)
            image_search_results = [None] * len(queries)
        # search for text
        if self.use_text_search and text_search_indices:
            text_search_results = self.searcher.text_search_agent(
                images, queries, text_search_indices, 
                image_rag_results=image_search_results, 
                reasoning=reasonings, 
                image_object_names=image_object_name, 
                image_search_indices=image_search_indices)
            text_search_queries = self.searcher.get_visit()["filtered_queries"]
            concated_queries = self.concat_search_queries(
                subqueries=text_search_queries, 
                queries=queries, 
                indices=text_search_indices
            )
            
        else:
            # If text search is disabled, ensure text_search_results are empty lists
            text_search_results = [[] for _ in range(len(queries))]
            concated_queries = queries
        text_concat_queries = [concated_queries[idx] for idx in indices]
        self.saver.save_str_list(text_concat_queries, prefix="rag_concated_queries", indices=indices)
        # replace None with empty lists to align output format.
        image_search_results = [result if result else [] for result in image_search_results]
        text_search_results  = [result if result else [] for result in text_search_results]
        search_results = [{"img_search" : image_search_results[idx], "text_search": text_search_results[idx]} for idx in range(len(queries))]
        rag_results = self.rag_context_builder.build(queries=concated_queries, images=images, search_results=search_results)
        rag_results = [rag_results[idx] for idx in indices]
        self.saver.save_str_list(rag_results, prefix="rag_results", indices=indices)
        responses, raw_responses = self.answer_agent(
                images=rag_images, 
                queries=rag_queries, 
                rag_ctx=rag_results, 
                msg_hist=[[]] * len(rag_images),
                return_prompts=False
            )
        answers, reasons = responses["answers"], responses["reasons"]
        verified_responses, raw_verified_responses = self.verifier(
            images=rag_images, 
            queries=rag_queries,
            evidences=rag_results, 
            answers=answers,
            reasons=reasons
        )
        
        rejects = self.verifier.register["rejects"]
        for idx, reject in enumerate(rejects):
            if reject:
                reasons[idx] = f"{reject} {reasons[idx]}"
        
        final_answers = verified_responses
        
        self.saver.save_str_list(answers, prefix="rag_answers", indices=indices)
        self.saver.save_str_list(reasons, prefix="rag_reasons", indices=indices)
        self.saver.save_str_list(raw_responses, prefix="rag_responses_raw", indices=indices)
        self.saver.save_str_list(verified_responses, prefix="rag_verifier", indices=indices)
        self.saver.save_str_list(raw_verified_responses, prefix="rag_verifier_raw", indices=indices)
        self.saver.save_str_list(final_answers, prefix="final_answers", indices=indices)
        return final_answers

    
    def text_verify_process(self, images, queries, reasonings, indices):
        return reasonings
        raise NotImplementedError("Text verify process is not implemented yet.")

    
    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        origin_images = images.copy()
        images = [resize_image(img, self.cfg.basic.max_img_size) for img in images]
        self.saver.save_str_list(queries, prefix="query")
        self.saver.save_image_list(images, prefix="image")
        # NOTE: do not use black list because it will cause the indices to be different.
        # TODO: align the indices in the answerable_agent.
        if self.use_black_list:
            _, black_list_indices, _ = self.query_filter(queries)
            while_list_indices = [i for i in range(len(queries)) if i not in black_list_indices]
            while_list_images = [images[i] for i in while_list_indices]
            while_list_queries = [queries[i] for i in while_list_indices]
        else:
            black_list_indices = []
            while_list_indices = list(range(len(queries)))
            while_list_images = images
            while_list_queries = queries
        domains = ['other'] * len(queries)
        if self.domain_aware:
            detected_domains = self.domain_router(while_list_images, while_list_queries)
            for idx, domain in zip(while_list_indices, detected_domains, strict=True):
                domains[idx] = domain
            self.saver.save_str_list(domains, prefix="domain")
        reasoning = ['I don\'t know' for _ in range(len(queries))]
        raw_reasoning = ['Black list: I don\'t know' for _ in range(len(queries))]
        while_list_reasoning, while_list_raw_reasoning = self.pre_answer_agent(while_list_images, while_list_queries, domains)
        for index, r, raw_r in zip(while_list_indices, while_list_reasoning, while_list_raw_reasoning, strict=True):
            reasoning[index] = r
            raw_reasoning[index] = raw_r
        flags, indices = self.answerable_agent(images, queries, reasoning, domains = domains)
        self.register["answerable_router_flag"] = flags
        self.saver.save_str_list(reasoning, prefix="reasoning")
        self.saver.save_str_list(raw_reasoning, prefix="reasoning_raw")
        self.saver.save_str_list(flags, prefix="answerable_router_flag")
        idk_indices, need_rag_indices, text_verify_indices, direct_output_indices = indices['idk'], indices['need_rag'], indices['text_verify'], indices['direct_output']
        assert len(idk_indices) + len(direct_output_indices) + len(text_verify_indices) + len(need_rag_indices) == len(while_list_queries), f"There are some indices in multiple branches {len(idk_indices) = }, {len(direct_output_indices) = }, {len(text_verify_indices) = }, {len(need_rag_indices) = }, {len(while_list_queries) = }."
        
        # Branch #1: Go to rag step.
        if need_rag_indices:
            # NOTE: Finished
            need_rag_answers = self.rag_process(images=images, queries=queries, reasonings=reasoning, indices=need_rag_indices, origin_images=origin_images)
        else:
            need_rag_answers = []

        # Branch #2: Direct output.
        if direct_output_indices:
            # NOTE: Finished
            direct_answers = [reasoning[idx] for idx in direct_output_indices]
            direct_queries = [queries[idx] for idx in direct_output_indices]
            direct_reasonings = [reasoning[idx] for idx in direct_output_indices]
            check_results = self.reasoning_answerable_checker(queries=direct_queries, reasonings=direct_reasonings)
            direct_answers = [direct_answer if check_result else "I don't know" for direct_answer, check_result in zip(direct_answers, check_results, strict=True)]
            direct_answers = [direct_answer if symbol not in direct_answer else "I don't know" for symbol in self.direct_black_list_symbols for direct_answer in direct_answers]
        else:
            direct_answers = []
            
        
        # Branch #3: Text verify.
        if text_verify_indices:
            # text_verify_answers = self.text_verify_process(images=images, queries=queries, reasonings=reasoning, indices=text_verify_indices)
            text_verify_answers = [reasoning[idx] for idx in text_verify_indices]
            text_verify_images = [images[idx] for idx in text_verify_indices]
            text_verify_queries = [queries[idx] for idx in text_verify_indices]
            verify_search_query_list, raw_verify_search_query_list = self.verify_text_search_generator(
                images=text_verify_images, 
                queries=text_verify_queries, 
                reasonings=text_verify_answers
            )
            enhanced_text_verify_generated_queries, raw_enhanced_text_verify_generated_queries = self.searcher.text_search_enhancer(
                images=text_verify_images, queries=text_verify_queries, subqueries=verify_search_query_list
            )
            filtered_queries, _ = self.searcher.filter_sub_questions(text_verify_queries, enhanced_text_verify_generated_queries, mode="hard")
            concated_queries = self.concat_search_queries(
                subqueries=filtered_queries, 
                queries=queries,
                indices=text_verify_indices
            )
            text_verified_concat_queries = [concated_queries[idx] for idx in text_verify_indices]
            self.saver.save_str_list(text_verified_concat_queries, prefix="text_verified_concat_queries", indices=text_verify_indices)
            self.saver.save_str_list(verify_search_query_list, prefix="search_queries", indices=text_verify_indices)
            self.saver.save_str_list(raw_verify_search_query_list, prefix="search_queries_raw", indices=text_verify_indices)
            self.saver.save_str_list(enhanced_text_verify_generated_queries, prefix="enhanced_text_search_queries", indices=text_verify_indices)
            self.saver.save_str_list(raw_enhanced_text_verify_generated_queries, prefix="enhanced_text_search_queries_raw", indices=text_verify_indices)
            verify_text_search_results = self.searcher.search(filtered_queries, search_type="text")
            self.saver.save_str_list(verify_text_search_results, prefix="text_search_results", indices=text_verify_indices)
            assert len(verify_text_search_results) == len(text_verify_indices)
            verify_search_result = [{"text_search": verify_text_search_results[idx]} for idx in range(len(text_verify_indices))]
            assert len(verify_search_result) == len(text_verify_indices) == len(text_verify_images) == len(text_verify_queries)
            text_verify_rag = self.rag_context_builder.build(
                queries=text_verified_concat_queries, 
                images=text_verify_images, 
                search_results=verify_search_result
            )
            assert len(text_verify_rag) == len(text_verify_indices)
            self.saver.save_str_list(text_verify_rag, prefix="rag_result", indices=text_verify_indices)
            text_verify_answers, raw_verified_responses = self.verifier(
                images=text_verify_images, 
                queries=text_verify_queries,
                evidences=text_verify_rag, 
                answers=text_verify_answers
            )
            self.saver.save_str_list(text_verify_answers, prefix="rag_verifier", indices=text_verify_indices)
            self.saver.save_str_list(raw_verified_responses, prefix="rag_verifier_raw", indices=text_verify_indices)
        else:
            text_verify_answers = []
            
        final_answers = ['I don\'t know' for _ in range(len(queries))]
        for i, direct_answer in zip(direct_output_indices, direct_answers, strict=True):
            final_answers[i] = direct_answer
        for i, text_verify_answer in zip(text_verify_indices, text_verify_answers, strict=True):
            final_answers[i] = text_verify_answer
        for i, need_rag_answer in zip(need_rag_indices, need_rag_answers, strict=True):
            final_answers[i] = need_rag_answer
        self.saver.save_str_list(final_answers, prefix="final_answers")
        self.saver.add_batch_cnt()
        return final_answers
            
        
class QADragon4Single(QADragon4Multi):
    def __init__(self, search_pipeline=None, enable_saver: bool = False, domain_aware: bool = False, all_text_verify: bool = True, use_black_list: bool = False, logging_level: str = "WARNING"):
        super().__init__(search_pipeline=search_pipeline, enable_saver=enable_saver, domain_aware=domain_aware, all_text_verify=all_text_verify, use_black_list=use_black_list, logging_level=logging_level)
        self.use_text_search = False
        
        
if __name__ == "__main__":
    import argparse
    from io import BytesIO
    from itertools import islice
    from urllib.request import urlopen

    import json
    from datasets import load_dataset
    from tqdm.contrib import tzip

    def parse_args():
        parser = argparse.ArgumentParser(description="Run qadragon Agent 2")
        parser.add_argument("--single", default=False, action="store_true", help="Enable single source agent")
        parser.add_argument("--saver", default=False, action="store_true", help="Enable saver")
        parser.add_argument("--domain_aware", default=False, action="store_true", help="Enable domain aware agent")
        parser.add_argument("--use_black_list", default=False, action="store_true", help="Enable black list")
        parser.add_argument("--llm_answerable", default=False, action="store_true", help="Enable rule answerable agent")
        parser.add_argument("--disable_lazy", default=False, action="store_true", help="Disable lazy loading of dataset")
        parser.add_argument("--all_text_verify", default=False, action="store_true", help="Enable all text verify")
        parser.add_argument("--no_search", default=False, action="store_true", help="Disable search")
        parser.add_argument("--logging_level", default="WARNING", type=str, help="Set logging level",
                            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        return parser.parse_args()
    
    # ------ Parse arguments
    args = parse_args()

    # ------ Config
    config_file = "configs/qadragon.yaml"
    cfg = OmegaConf.load(config_file)
    batch_size = cfg.basic.batch_size
    search_api_text_model = "BAAI/bge-large-en-v1.5"
    search_api_image_model = "openai/clip-vit-large-patch14-336"
    web_index = "crag-mm-2025/web-search-index-validation"
    image_index = "crag-mm-2025/image-search-index-validation"

    # ------ Build pipeline & agent
    if args.no_search:
        search_pipeline = None
    else:
        search_pipeline = UnifiedSearchPipeline(
            text_model_name=search_api_text_model,
            image_model_name=search_api_image_model,
            web_hf_dataset_id=web_index,
            image_hf_dataset_id=image_index,
        )
    
    if args.single:
        agent = QADragon4Single(search_pipeline=search_pipeline, enable_saver=args.saver, domain_aware=args.domain_aware, logging_level=args.logging_level, all_text_verify=args.all_text_verify, use_black_list=args.use_black_list)
    else:
        agent = QADragon4Multi(search_pipeline=search_pipeline, enable_saver=args.saver, domain_aware=args.domain_aware, logging_level=args.logging_level, all_text_verify=args.all_text_verify, use_black_list=args.use_black_list)

    # ------ Load dataset with or without streaming    
    dataset = load_dataset(
        "crag-mm-2025/crag-mm-single-turn-public",
        revision="v0.1.2",
        split="validation",
        streaming= not args.disable_lazy,
    )

    def batch_iterator(dataset, batch_size):
        it = iter(dataset)
        while True:
            batch = list(islice(it, batch_size))
            if not batch:
                break
            # Convert list of dicts to dict of lists
            batch_dict = {}
            if batch:
                for key in batch[0]:
                    batch_dict[key] = [item[key] for item in batch]
            yield batch_dict

    # ------ Open output file (if needed)
    output_file = None
    if cfg.saver.enabled:
        # save config
        with open(os.path.join(agent.saver.get_save_dir(), "qadragon.yaml"), "w") as f:
            OmegaConf.save(config=cfg, f=f)
            
        output_file = open(os.path.join(agent.saver.get_save_dir(), "output.jsonl"), "w")
        output_file.write("[\n")
    # ------ Batch loop
    global_idx = 0
    for batch_index, batch_data in enumerate(batch_iterator(dataset, batch_size) ):
        time_start = time.time()
        # prepare data
        queries = [turn["query"][0] for turn in batch_data["turns"]]
        images = batch_data["image"]
        image_urls = batch_data["image_url"]
        expected = [ans["ans_full"][0] for ans in batch_data["answers"]]
        images = repair_images(images, image_urls, batch_index * batch_size, config=cfg)
        hist_batch = [[]] * len(queries)
        agent.saver.set_batch_size(len(queries))
        # generate response
        preds = agent.batch_generate_response(queries, images, hist_batch)
        # output batch time.
        time_end = time.time()
        time_cost = round(time_end - time_start, 2)
        print(f"Batch {global_idx // batch_size} time: {time_cost}s")
        if time_cost > 10 * batch_size:
            print(f">>>>>>>>>>>>>>>>>>> Batch {global_idx // batch_size} Out of Time Warning <<<<<<<<<<<<<<<<<<<<")
        if output_file:
            for i, (q, p, exp, flag) in enumerate(zip(queries, preds, expected, agent.register["answerable_router_flag"], strict=True)):
                output_file.write(
                    json.dumps(
                        {"idx": global_idx, "query": q, "prediction": p, "gold": exp, "answerable_router_flag": flag},
                        indent=4,
                    )
                    + ",\n"
                )
                global_idx += 1
                output_file.flush()
        else:
            print(
                f"queries: {queries}, "
                f"predictions: {preds}, expected: {expected}"
            )

    if output_file:
        output_file.seek(output_file.tell() - 2, 0)  # Move back 2 characters
        output_file.truncate()  # Remove the trailing ',\n'
        output_file.write("\n]")
        output_file.close()