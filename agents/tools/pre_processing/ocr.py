import easyocr
from spellchecker import SpellChecker
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import numpy as np
import textwrap
from PIL import Image
import vllm
import json_repair
import re
from typing import List


class BaseOCRProcessor:
    def __init__(self, languages=["en"], confidence_threshold=0.6):
        pass

    def process_image(self, query, image):
        return ""

    def ocr_easyocr(self, query, image):
        return ""


class OCRProcessor(BaseOCRProcessor):
    def __init__(
        self,
        languages=["en"],
        confidence_threshold=0.6,
        ocr_type="easyocr",
        LLM=None,
        tokenizer=None,
        config=None,
    ):
        self.languages = languages
        self.config = config
        self.spell_checker = SpellChecker()
        self.confidence_threshold = confidence_threshold
        self.ocr_type = ocr_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if ocr_type == "GOT-OCR":
            self.model = AutoModelForImageTextToText.from_pretrained(
                "stepfun-ai/GOT-OCR-2.0-hf", device_map=self.device
            )
            self.processor = AutoProcessor.from_pretrained(
                "stepfun-ai/GOT-OCR-2.0-hf", use_fast=True
            )
            self.reader = None
        elif ocr_type == "easyocr":
            self.reader = easyocr.Reader(self.languages)
            self.model = None
            self.processor = None
        elif ocr_type == "vllm":
            if LLM is None:
                raise ValueError("LLM is not provided")
            if tokenizer is None:
                raise ValueError("Tokenizer is not provided")
            self.LLM = LLM
            self.tokenizer = tokenizer
        else:
            raise ValueError(
                "Invalid OCR type. Choose either 'easyocr', 'GOT-OCR', or 'vllm'."
            )

    def process_image(self, query, image, seg_obj: str):
        if self.ocr_type == "easyocr":
            results = self.reader.readtext(
                image,
                decoder="greedy",
                detail=1,
                paragraph=True,
                threshold=self.confidence_threshold,
            )
            corrected_result = ""
            for word in results:
                corrected_word = self.spell_checker.correction(word[1])
                if corrected_word == None:
                    corrected_word = word[1]
                corrected_result += corrected_word + " "
        elif self.ocr_type == "GOT-OCR":
            pil_image = Image.fromarray(image)
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)

            generate_ids = self.model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self.processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=4096,
            )

            result = self.processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            corrected_result = " ".join(
                [
                    (
                        self.spell_checker.correction(word)
                        if self.spell_checker.correction(word) is not None
                        else word
                    )
                    for word in result.split()
                ]
            )
            return corrected_result
        elif self.ocr_type == "vllm":
            image = Image.fromarray(image)
            ocr_prompt = {
                "prompt": self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": textwrap.dedent(
                                f"""
                        You are an OCR assistant. Extract the text on the image based on user's instruction. Do not explain or answer the question. 
                        Return text in the JSON format: {{"text": "<related text>"}} or {{"text": null}} if nothing relevant is found.
                    """
                            ),
                        },
                        {"role": "user", "content": [{"type": "image"}]},
                        {
                            "role": "user",
                            "content": textwrap.dedent(
                                f"""\
                            Conduct OCR on the image to scan all the text in your mind.
                            Then, extract the relevant text in the image especially around the "{seg_obj}" to answer the question "{query}".
                            Return the text in the JSON format {{"text": "<related text>"}}.
                            If nothing relevant, return {{"text": null}}.
                            Do not answer the question. 
                    """
                            ),
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ),
                "multi_modal_data": {"image": image},
            }

        outputs = self.LLM.generate(
            [ocr_prompt],
            sampling_params=vllm.SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=self.config.tools.ocr.config.max_tokens,
                skip_special_tokens=True,
            ),
        )
        ocr = outputs[0].outputs[0].text
        print(f"OCR output:", ocr)

        def try_parse_ocr(text):
            try:
                match = re.search(r"\{.*?\}", text, re.DOTALL)
                if match:
                    ocr_json = json_repair.loads(match.group(0))
                    if "text" in ocr_json and ocr_json["text"]:
                        return ocr_json["text"]
            except Exception:
                pass
            return ""

        ocr = try_parse_ocr(ocr)

        # If not valid and reformat_with_llm, ask LLM to reformat as JSON
        if not ocr and self.config.tools.ocr.config.reformat_with_llm:
            reformat_prompt = {
                "prompt": self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that reformats text into valid JSON according to a given schema.",
                        },
                        {
                            "role": "user",
                            "content": textwrap.dedent(
                                f"""
                                The following is an attempt to extract text, but it may not be valid JSON:
                                {ocr}
                                Please reformat it into valid JSON in the following format:
                                {{"text": "<related text>"}}
                            """
                            ),
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            }
            outputs = self.LLM.generate(
                [reformat_prompt],
                sampling_params=vllm.SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=self.config.tools.ocr.config.max_tokens,
                    skip_special_tokens=True,
                ),
            )
            reformatted = outputs[0].outputs[0].text
            print("reformat output:", reformatted)
            ocr = try_parse_ocr(reformatted)
            if ocr is None or ocr == "null":
                ocr = ""
        return [ocr]

    def batch_process_images(self, query_list, images, seg_obj_list):
        results = []
        for query, image, seg_obj in zip(query_list, images, seg_obj_list):
            result = self.process_image(query, image, seg_obj)
            results.append(result)
        return results


if __name__ == "__main__":
    ocr_processor = OCRProcessor()
    image_path = "example/example.jpg"  # Replace with your image path
    image = np.array(Image.open(image_path).convert("RGB"))
    text = ocr_processor.process_image(image)
    print(text)
