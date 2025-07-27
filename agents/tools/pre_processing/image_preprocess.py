from PIL import Image

import torch
import vllm
import os
import sys

# from agents.tools.pre_processing.super_resolution import SuperResolution
sys.path.append(".")
from agents.tools.pre_processing.image_caption import ImageCaption
from agents.tools.pre_processing.main_word_extract import WordExtractor
from agents.tools.pre_processing.segment import ImageSegmenter
from agents.tools.pre_processing.ocr import OCRProcessor
from typing import List, Tuple
import numpy as np
from omegaconf import OmegaConf


class ImagePreprocessor:
    def __init__(self, llm: object = None, tokenizer: object = None, config=None):
        self.llm = llm
        self.tokenizer = tokenizer
        # self.captioner = ImageCaption(mode=config.tools.captioning.config.mode, LLM=self.llm, tokenizer=self.tokenizer)\
        self.is_seg = config.tools.segmentation.enabled
        self.is_ocr = config.tools.ocr.enabled
        self.is_caption = config.tools.captioning.enabled
        self.is_word_extract = config.tools.keyword_extraction.enabled
        self.is_preprocess = config.enabled
        if self.is_seg:
            self.segmenter = ImageSegmenter(config=config)
        else:
            self.segmenter = None
        if self.is_word_extract:
            self.word_extractor = WordExtractor(
                LLM=self.llm,
                tokenizer=self.tokenizer,
                ext_type=config.tools.keyword_extraction.config.ext_type,
                config=config,
            )
        else:
            self.word_extractor = None
        if self.is_caption:
            self.image_caption = ImageCaption(
                mode=config.tools.captioning.config.mode,
                LLM=self.llm,
                tokenizer=self.tokenizer,
                config=config,
            )
        else:
            self.image_caption = None
        if self.is_ocr:
            self.ocr_processor = OCRProcessor(
                languages=config.tools.ocr.config.languages,
                confidence_threshold=config.tools.ocr.config.confidence_threshold,
                ocr_type=config.tools.ocr.config.ocr_type,
                LLM=self.llm,
                tokenizer=self.tokenizer,
                config=config,
            )
        else:
            self.ocr_processor = None

    def preprocess_image(
        self, image: Image.Image, query: str
    ) -> Tuple[dict, List[str]]:
        if self.is_preprocess == True:
            if self.is_word_extract:
                nouns = self.word_extractor.extract_nouns(image, query)
                key_words = nouns
                if len(key_words) == 0:
                    key_words = [query]
                    # key_words = ["mainObject"]
            else:
                key_words = [query]
            if self.is_seg and image is not None:
                seg_image_list = self.segmenter.segment_image(image, names=key_words)
                if seg_image_list is None:
                    names = ["mainObject"]
                    additional_seg_image_list = self.segmenter.simple_segment_image(
                        image, names=names
                    )
                    if additional_seg_image_list is None:
                        seg_image_list = {}
                    else:
                        seg_image_list = additional_seg_image_list
            else:
                seg_image_list = {}

            if seg_image_list != {}:
                for category, images in seg_image_list.items():
                    for img_dict in images:
                        img = img_dict["img"]
                        if self.is_ocr:
                            ocr_result = self.ocr_processor.process_image(
                                query=query, image=np.array(img), seg_obj=category
                            )
                        else:
                            ocr_result = None
                        if self.is_caption:
                            caption = self.image_caption.single_generate_caption(img)
                        else:
                            caption = None

                        if ocr_result:
                            img_dict["ocr_result"] = ocr_result
                        else:
                            img_dict["ocr_result"] = ""
                        if caption:
                            img_dict["caption"] = caption
                        else:
                            img_dict["caption"] = ""
            if self.is_ocr and image is not None:
                entire_ocr_result = self.ocr_processor.process_image(
                    query=query, image=np.array(image), seg_obj="entire image"
                )
            else:
                entire_ocr_result = None
            if self.is_caption:
                entire_caption = self.image_caption.single_generate_caption(image)
            else:
                entire_caption = ""

            if entire_ocr_result:
                seg_image_list["origin_image"] = [
                    {
                        "img": image,
                        "ocr_result": entire_ocr_result,
                        "caption": entire_caption,
                    }
                ]
            else:
                seg_image_list["origin_image"] = [
                    {"img": image, "ocr_result": "", "caption": entire_caption}
                ]
            return seg_image_list, key_words
        else:
            seg_image_list = {
                "origin_image": [{"img": image, "ocr_result": "", "caption": ""}]
            }
            key_words = [query]
            return seg_image_list, key_words

    def batch_preprocess_images(
        self, images: List[Image.Image], queries: List[str], version: str = "v2"
    ) -> Tuple[List[dict], List[List[str]]]:
        results = []
        key_words_list = []
        key_words_list = self.word_extractor.batch_extract_nouns(images, queries)
        results = self.segmenter.batch_segment_images(images, names=key_words_list, version=version)
        # Check if results are empty and add the original image if necessary
        if version == "v1":
            for i, result in enumerate(results):
                if result == {}:
                    results[i] = {"origin_image": [{"img": images[i], "score": 0}]}
        elif version == "v2":
            for i, result in enumerate(results):
                if result is None:
                    results[i] = images[i]
        return results, key_words_list


if __name__ == "__main__":
    image_path = "docs/dataset_info/singleqa.jpg"
    querys = ["what is the cost of this scooter?", "what is the scooter cost?"]
    config = OmegaConf.load("configs/basic.yaml").preprocessing
    images = [Image.open(image_path), Image.open(image_path)]
    # llm Configuration
    LLM_CONFIG = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 8192,
        "max_num_seqs": 2,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "limit_mm_per_prompt": {"image": 1},
    }

    # Keyword Extraction Configuration
    KEYWORD_CONFIG = {
        "temperature": 0.7,
        "top_p": 0.5,
        "max_tokens": 200,
        "top_k": 3,
        "max_retries": 2,
    }
    # Initialize vLLM
    llm = vllm.LLM(model="meta-llama/Llama-3.2-11B-Vision-Instruct", **LLM_CONFIG)
    tokenizer = llm.get_tokenizer()
    preprocessor = ImagePreprocessor(llm=llm, tokenizer=tokenizer, config=config)
    result, keywords_list = preprocessor.batch_preprocess_images(images, querys)
    print("Preprocessing result:")
    print(result)
    print("Keywords list:")
    print(keywords_list)
    # Output example:
    # Preprocessing result:
    # {'scooter cost': [{'img': <PIL.Image.Image image mode=RGB size=115x82 at 0x7BA0345C3970>, 'score': np.float32(0.45852026), 'ocr_result': '', 'caption': 'a picture of a red motorcycle with a black seat and kickstand is parked on the street next to a blue car, with a white line marking the edge of the parking space'}], 'origin_image': [{'img': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=277x369 at 0x7BA0344E6E00>, 'ocr_result': '', 'caption': 'a picture of a man in a black coat stands next to a red motorcycle parked in front of a tall building, with a leafless tree in the background and a clear blue sky above'}]}
    # Keywords list:
    # ['Scooter', 'Cost']
