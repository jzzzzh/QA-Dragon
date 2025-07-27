import re
import textwrap

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import vllm
from typing import List, Dict, Any
from PIL import Image
import json_repair


class WordExtractor:
    def __init__(self, LLM=None, tokenizer=None, ext_type="tech", config=None):
        self.config = config
        self.ext_type = ext_type
        if ext_type == "tech":
            # Load model directly
            self.tokenizer = AutoTokenizer.from_pretrained(
                "ilsilfverskiold/tech-keywords-extractor"
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                "ilsilfverskiold/tech-keywords-extractor"
            )
        elif ext_type == "vllm" or ext_type == "step_vllm":
            if LLM is None:
                raise ValueError("LLM is not provided")
            if tokenizer is None:
                raise ValueError("Tokenizer is not provided")
            self.LLM = LLM
            self.tokenizer = tokenizer
        # Initialize lists for pronouns
        self.pronoun = ["this?", "that?", "these?", "those?", "it?"]
        self.person_pronoun = [
            "I",
            "me",
            "we",
            "us",
            "he",
            "him",
            "she",
            "they",
            "them",
        ]
        self.top_k = config.tools.keyword_extraction.config.top_k

    def extract_nouns(self, img, sentence):
        if self.ext_type == "tech":
            input_sequences = sentence
            input_ids = self.tokenizer(
                input_sequences, return_tensors="pt", truncation=True
            ).input_ids
            output = self.model.generate(input_ids, no_repeat_ngram_size=3, num_beams=4)
            predicted = self.tokenizer.decode(output[0], skip_special_tokens=True)
            key_words = [word.strip() for word in predicted.split(",") if word.strip()]

            # Check for pronouns and add a placeholder if found
            try:
                words_in_sentence = sentence.lower().split()
            except Exception as e:
                words_in_sentence = ["mainObject"]
            if any(pronoun in words_in_sentence for pronoun in self.pronoun):
                key_words.append("mainObject")
            if any(pronoun in words_in_sentence for pronoun in self.person_pronoun):
                key_words.append("mainPerson")
            if len(key_words) > self.top_k:
                key_words = key_words[: self.top_k]
            return key_words
        elif self.ext_type == "vllm":
            prompt = {
                "prompt": self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "You are an expert AI system for object detection and identification. Your goal is to recognize objects in an image and determine the one most relevant to answering the given question. Do not include any explanations or additional text. Only return structured results in JSON format.",
                        },
                        {"role": "user", "content": [{"type": "image"}]},
                        {
                            "role": "user",
                            "content": textwrap.dedent(
                                f"""
                                First, list up to 3 distinct major objects in the image that are relevant to the question: "{sentence}". 
                                Each object name should be concise (no more than 3 words) and refer to a tangible item  (e.g., "brand", "building", "car", "dog", "screen", "book", "plant", "food", "paint").
                                For some specific objects, you should change the object to its category name, like specific brand name to "brand", specific car brand to "car", etc.
                                Return the object list in the following JSON format:
                                {{"all_objects": ["<object_name>", "<object_name>", ...]}}

                                Next, identify the **single object** in the image that the question "{sentence}" specifically refers to. 
                                Pay close attention to word spacing and phrasing in the question to ensure correct understanding.

                                Return only the final selected object in this format:
                                {{"object": "<object_name>"}}
                            """
                            ),
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ),
                "multi_modal_data": {"image": img},
            }

            retries = 0
            while retries <= self.config.tools.keyword_extraction.config.max_retries:
                outputs = self.LLM.generate(
                    [prompt],
                    sampling_params=vllm.SamplingParams(
                        temperature=self.config.tools.keyword_extraction.config.llm_temperature,
                        top_p=self.config.tools.keyword_extraction.config.llm_top_p,
                        max_tokens=self.config.tools.keyword_extraction.config.max_tokens,
                        skip_special_tokens=True,
                    ),
                )

                predicted = outputs[0].outputs[0].text
                print(f"Attempt {retries + 1} output:", predicted)

                try:
                    # Try to match complete JSON object first
                    full_matches = re.findall(r"\{.*?\}", predicted)
                    if full_matches:
                        keywords = [
                            json_repair.loads(match.strip()) for match in full_matches
                        ]
                        keywords = [
                            keyword["object"]
                            for keyword in keywords
                            if "object" in keyword
                        ]
                        if keywords:  # If we found valid keywords, return them
                            return keywords[
                                : self.config.tools.keyword_extraction.config.top_k
                            ]

                    # Try to match partial JSON starting with {
                    partial_matches = re.findall(r"\{(.*)", predicted)
                    if partial_matches:
                        keywords = [
                            json_repair.loads(match.strip()) for match in full_matches
                        ]
                        keywords = [
                            keyword["object"]
                            for keyword in keywords
                            if "object" in keyword
                        ]
                        if keywords:  # If we found valid keywords, return them
                            return keywords[
                                : self.config.tools.keyword_extraction.config.top_k
                            ]

                    retries += 1
                    if (
                        retries
                        <= self.config.tools.keyword_extraction.config.max_retries
                    ):
                        print(
                            f"No valid keywords found, retrying... (Attempt {retries + 1}/{self.config.tools.keyword_extraction.config.max_retries + 1})"
                        )
                except Exception:
                    print(
                        f"Match error, retrying... (Attempt {retries + 1}/{self.config.tools.keyword_extraction.config.max_retries + 1})"
                    )
                    continue

            # If we've exhausted all retries and still no match, return default
            print("No valid keywords found after all retries, using default")
            return ["mainObject"]
        elif self.ext_type == "step_vllm":
            """Stepwise extraction: first list objects, then select the main one. Optionally reformat with LLM if output is not valid JSON."""
            # Step 1: List up to 3 major objects
            list_prompt = {
                "prompt": self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "You are an expert AI system for object detection and identification. Your goal is to recognize objects in an image relevant to a question. Only return structured results in JSON format.",
                        },
                        {"role": "user", "content": [{"type": "image"}]},
                        {
                            "role": "user",
                            "content": textwrap.dedent(
                                f"""
                                List up to 3 distinct major objects in the image that are relevant to the question: "{sentence}". 
                                Each object name should be concise (no more than 3 words) and refer to a tangible item (e.g., "brand", "building", "car", "dog", "screen", "book", "plant", "food", "paint").
                                Replace specific instances with general categories (e.g., "ZARA" → "brand", "BMW" → "car", "cook guidance" → "book").
                                Return the object list in the following JSON format:
                                {{"all_objects": ["<object_name>", "<object_name>", ...]}}
                            """
                            ),
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ),
                "multi_modal_data": {"image": img},
            }

            outputs = self.LLM.generate(
                [list_prompt],
                sampling_params=vllm.SamplingParams(
                    temperature=self.config.tools.keyword_extraction.config.llm_temperature,
                    top_p=self.config.tools.keyword_extraction.config.llm_top_p,
                    max_tokens=self.config.tools.keyword_extraction.config.max_tokens,
                    skip_special_tokens=True,
                ),
            )
            predicted = outputs[0].outputs[0].text
            print(f"Step 1 (list objects) output:", predicted)
            all_objects = None

            def try_parse_all_objects(text):
                try:
                    match = re.search(r"\{.*?\}", text, re.DOTALL)
                    if match:
                        obj_json = json_repair.loads(match.group(0))
                        if (
                            "all_objects" in obj_json
                            and isinstance(obj_json["all_objects"], list)
                            and obj_json["all_objects"]
                        ):
                            return obj_json["all_objects"][
                                : self.config.tools.keyword_extraction.config.top_k
                            ]
                except Exception:
                    pass
                return None

            all_objects = try_parse_all_objects(predicted)

            # If not valid and reformat_with_llm, ask LLM to reformat as JSON
            if (
                not all_objects
                and self.config.tools.keyword_extraction.config.reformat_with_llm
            ):
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
                                    The following is an attempt to list objects in an image, but it may not be valid JSON:
                                    {predicted}
                                    Please reformat it into valid JSON in the following format:
                                    {{"all_objects": ["<object_name>", "<object_name>", ...]}}
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
                        max_tokens=self.config.tools.keyword_extraction.config.max_tokens,
                        skip_special_tokens=True,
                    ),
                )
                reformatted = outputs[0].outputs[0].text
                print("Step 1 (reformat) output:", reformatted)
                all_objects = try_parse_all_objects(reformatted)

            if not all_objects:
                print("No valid object list found, using default")
                return ["mainObject"]

            # jump out if there is a screen or monitor, because it is generally shown in the screen and LLM can not understand it.
            skip_words = [
                "book",
                "screen",
                "monitor",
            ]  # note that there is an order for the skip words.
            for object_ in all_objects:
                if not isinstance(object_, str):
                    try:
                        object_ = object_.get("object", "")
                    except Exception:
                        object_ = ""
                for skip_word in skip_words:
                    if skip_word in object_.lower():
                        return [skip_word]

            # Step 2: Ask for the single most relevant object
            select_prompt = {
                "prompt": self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "You are an expert AI system for object detection and identification. Only return structured results in JSON format.",
                        },
                        {"role": "user", "content": [{"type": "image"}]},
                        {
                            "role": "user",
                            "content": textwrap.dedent(
                                f"""
                                Given the following list of objects detected in the image: {all_objects}
                                and the question: "{sentence}"
                                Identify the **single object** in the image that the question specifically refers to.
                                Pay close attention to spatial information in the question to ensure correct understanding.
                                Return only the final selected object in this format:
                                {{"object": "<object_name>"}}
                            """
                            ),
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ),
                "multi_modal_data": {"image": img},
            }

            outputs = self.LLM.generate(
                [select_prompt],
                sampling_params=vllm.SamplingParams(
                    temperature=self.config.tools.keyword_extraction.config.llm_temperature,
                    top_p=self.config.tools.keyword_extraction.config.llm_top_p,
                    max_tokens=self.config.tools.keyword_extraction.config.max_tokens,
                    skip_special_tokens=True,
                ),
            )
            predicted = outputs[0].outputs[0].text
            print(f"Step 2 (select object) output:", predicted)
            selected_object = None

            def try_parse_selected_object(text):
                try:
                    match = re.search(r"\{.*?\}", text, re.DOTALL)
                    if match:
                        obj_json = json_repair.loads(match.group(0))
                        if "object" in obj_json and obj_json["object"]:
                            return obj_json["object"]
                except Exception:
                    pass
                return None

            selected_object = try_parse_selected_object(predicted)

            # If not valid and reformat_with_llm, ask LLM to reformat as JSON
            if (
                not selected_object
                and self.config.tools.keyword_extraction.config.reformat_with_llm
            ):
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
                                    The following is an attempt to select an object, but it may not be valid JSON:
                                    {predicted}
                                    Please reformat it into valid JSON in the following format:
                                    {{"object": "<object_name>"}}
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
                        max_tokens=self.config.tools.keyword_extraction.config.max_tokens,
                        skip_special_tokens=True,
                    ),
                )
                reformatted = outputs[0].outputs[0].text
                print("Step 2 (reformat) output:", reformatted)
                selected_object = try_parse_selected_object(reformatted)

            if not selected_object or selected_object == "None":
                print("No valid object found, using default")
                selected_object = "mainObject"

            return [selected_object]


    def extract_keywords_step(self, imgs, sentences):
        """Stepwise extraction: first list objects, then select the main one. Optionally reformat with LLM if output is not valid JSON."""
        # Step 1: List up to 3 major objects
        list_prompt = []
        for img, sentence in zip(imgs, sentences):
            prompt = {
                "prompt": self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "You are an expert AI system for object detection and identification. Your goal is to recognize objects in an image relevant to a question. Only return structured results in JSON format.",
                        },
                        {"role": "user", "content": [{"type": "image"}]},
                        {
                            "role": "user",
                            "content": textwrap.dedent(
                                f"""
                                List up to 3 distinct major objects in the image that are relevant to the question: "{sentence}". 
                                Each object name should be concise (no more than 3 words) and refer to a tangible item (e.g., "brand", "building", "car", "dog", "screen", "book", "plant", "food", "paint").
                                Replace specific instances with general categories (e.g., "ZARA" → "brand", "BMW" → "car", "cook guidance" → "book").
                                Return the object list in the following JSON format:
                                {{"all_objects": ["<object_name>", "<object_name>", ...]}}
                            """
                            ),
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ),
                "multi_modal_data": {"image": img},
            }
            list_prompt.append(prompt)
        outputs = self.LLM.generate(
            list_prompt,
            sampling_params=vllm.SamplingParams(
                temperature=self.config.tools.keyword_extraction.config.llm_temperature,
                top_p=self.config.tools.keyword_extraction.config.llm_top_p,
                max_tokens=self.config.tools.keyword_extraction.config.max_tokens,
                skip_special_tokens=True,
            ),
        )
        predicted_list = [output.outputs[0].text.strip() for output in outputs]
        print(f"Step 1 (list objects) {len(predicted_list)} output:")
        for predicted in predicted_list:
            print(predicted)    
            print("=" * 20)
        all_objects = None

        def try_parse_all_objects(text):
            try:
                match = re.search(r"\{.*?\}", text, re.DOTALL)
                if match:
                    obj_json = json_repair.loads(match.group(0))
                    if (
                        "all_objects" in obj_json
                        and isinstance(obj_json["all_objects"], list)
                        and obj_json["all_objects"]
                    ):
                        return obj_json["all_objects"][
                            : self.config.tools.keyword_extraction.config.top_k
                        ]
            except Exception:
                pass
            return None

        all_objects_list = [try_parse_all_objects(predicted) for predicted in predicted_list]
        print("Step 1 (try_parse_all_objects) output:")
        for i, all_objects in enumerate(all_objects_list):
            print(f"Image {i}: {all_objects}")
        
        # If not valid and reformat_with_llm, ask LLM to reformat as JSON
        is_final_list = []
        reformat_prompt_list = []
        step1_reformat_prompt_cnt_list = []
        for i, (all_objects, predicted) in enumerate(zip(all_objects_list, predicted_list)):
            if (
                not all_objects
                and self.config.tools.keyword_extraction.config.reformat_with_llm
            ):
                is_final_list.append(False)
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
                                    The following is an attempt to list objects in an image, but it may not be valid JSON:
                                    {predicted}
                                    Please reformat it into valid JSON in the following format:
                                    {{"all_objects": ["<object_name>", "<object_name>", ...]}}
                                """
                                ),
                            },
                        ],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                }
                reformat_prompt_list.append(reformat_prompt)
                step1_reformat_prompt_cnt_list.append(i)
            else:
                is_final_list.append(True)
                
        step2_reformat_prompt_cnt_list = []   
        reformatted_list = []
        if len(reformat_prompt_list) > 0:
            outputs = self.LLM.generate(
                reformat_prompt_list,
                sampling_params=vllm.SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=self.config.tools.keyword_extraction.config.max_tokens,
                    skip_special_tokens=True,
                ),
            )
            reformatted_list = [output.outputs[0].text.strip() for output in outputs]
            print("Step 1 (reformat) output:", reformatted_list)
            tmp_all_objects_list = [try_parse_all_objects(reformatted) for reformatted in reformatted_list]
        if len(reformat_prompt_list) > 0:
            for i, (all_objects, reformatted) in enumerate(
                zip(tmp_all_objects_list, reformatted_list)
            ):
                if (
                    not all_objects
                    and self.config.tools.keyword_extraction.config.reformat_with_llm
                ):
                    skip_words = [
                        "book",
                        "screen",
                        "monitor",
                    ]  # note that there is an order for the skip words.
                    for object_ in all_objects:
                        if not isinstance(object_, str):
                            try:
                                object_ = object_.get("object", "")
                            except Exception:
                                object_ = ""
                        for skip_word in skip_words:
                            if skip_word in object_.lower():
                                all_objects = [skip_word]
                                is_final_list[step1_reformat_prompt_cnt_list[i]] = True
                            else:
                                step2_reformat_prompt_cnt_list.append(
                                    step1_reformat_prompt_cnt_list[i]
                                )
                    all_objects_list[step1_reformat_prompt_cnt_list[i]] = all_objects
                    predicted_list[step1_reformat_prompt_cnt_list[i]] = reformatted
                    print(f"Image {step1_reformat_prompt_cnt_list[i]}: {all_objects}")
                else:
                    is_final_list[step1_reformat_prompt_cnt_list[i]] = True
                    all_objects_list[step1_reformat_prompt_cnt_list[i]] = all_objects
                    predicted_list[step1_reformat_prompt_cnt_list[i]] = reformatted
                    print(f"Image {step1_reformat_prompt_cnt_list[i]}: {all_objects}")
                    
                if not all_objects:
                    print("No valid object list found, using default")
                    all_objects_list[step1_reformat_prompt_cnt_list[i]] = ["mainObject"]
                    predicted_list[step1_reformat_prompt_cnt_list[i]] = reformatted
                    is_final_list[step1_reformat_prompt_cnt_list[i]] = True
            

        # jump out if there is a screen or monitor, because it is generally shown in the screen and LLM can not understand it.
        select_prompt_list = [] 
        # Step 2: Ask for the single most relevant object
        for idx, (is_final, img, sentence, all_objects) in enumerate(zip(
            is_final_list, imgs, sentences, all_objects_list
        )):
            if not is_final:
                select_prompt = {
                    "prompt": self.tokenizer.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": "You are an expert AI system for object detection and identification. Only return structured results in JSON format.",
                            },
                            {"role": "user", "content": [{"type": "image"}]},
                            {
                                "role": "user",
                                "content": textwrap.dedent(
                                    f"""
                                    Given the following list of objects detected in the image: {all_objects}
                                    and the question: "{sentence}"
                                    Identify the **single object** in the image that the question specifically refers to.
                                    Pay close attention to spatial information in the question to ensure correct understanding.
                                    Return only the final selected object in this format:
                                    {{"object": "<object_name>"}}
                                """
                                ),
                            },
                        ],
                        add_generation_prompt=True,
                        tokenize=False,
                    ),
                    "multi_modal_data": {"image": img},
                }
                select_prompt_list.append(select_prompt)
                
        outputs = self.LLM.generate(
            select_prompt_list,
            sampling_params=vllm.SamplingParams(
                temperature=self.config.tools.keyword_extraction.config.llm_temperature,
                top_p=self.config.tools.keyword_extraction.config.llm_top_p,
                max_tokens=self.config.tools.keyword_extraction.config.max_tokens,
                skip_special_tokens=True,
            ),
        )
        predicted_list = [output.outputs[0].text.strip() for output in outputs]
        print(f"Step 2 (select object) output:", predicted_list)
        selected_object_list = []

        def try_parse_selected_object(text):
            try:
                match = re.search(r"\{.*?\}", text, re.DOTALL)
                if match:
                    obj_json = json_repair.loads(match.group(0))
                    if "object" in obj_json and obj_json["object"]:
                        return obj_json["object"]
            except Exception:
                pass
            return None

        selected_object_list = [try_parse_selected_object(predicted) for predicted in predicted_list]
        reformat_prompt_list = []
        step3_reformat_prompt_cnt_list = []
        for i, (selected_object, predicted) in enumerate(
            zip(selected_object_list, predicted_list)
        ):
            # If not valid and reformat_with_llm, ask LLM to reformat as JSON
            if (
                not selected_object
                and self.config.tools.keyword_extraction.config.reformat_with_llm
            ):
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
                                    The following is an attempt to select an object, but it may not be valid JSON:
                                    {predicted}
                                    Please reformat it into valid JSON in the following format:
                                    {{"object": "<object_name>"}}
                                """
                                ),
                            },
                        ],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                }
                reformat_prompt_list.append(reformat_prompt)
                step3_reformat_prompt_cnt_list.append(step2_reformat_prompt_cnt_list[i])
            else:
                is_final_list[step2_reformat_prompt_cnt_list[i]] = True
                all_objects_list[step2_reformat_prompt_cnt_list[i]] = selected_object
                predicted_list[step2_reformat_prompt_cnt_list[i]] = predicted
                
            outputs = self.LLM.generate(
                reformat_prompt_list,
                sampling_params=vllm.SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=self.config.tools.keyword_extraction.config.max_tokens,
                    skip_special_tokens=True,
                ),
            )
            reformatted_list = [output.outputs[0].text.strip() for output in outputs]
            print("Step 2 (reformat) output:", reformatted_list)
            selected_object_list = [try_parse_selected_object(reformatted) for reformatted in reformatted_list]

        for idx, selected_object in enumerate(selected_object_list):
            if selected_object:
                all_objects_list[step3_reformat_prompt_cnt_list[idx]] = selected_object
                predicted_list[step3_reformat_prompt_cnt_list[idx]] = reformatted
                is_final_list[step3_reformat_prompt_cnt_list[idx]] = True
                print(f"Image {step3_reformat_prompt_cnt_list[idx]}: {selected_object}")
            else:
                print("No valid object found, using default")
                is_final_list[step3_reformat_prompt_cnt_list[idx]] = True
                all_objects_list[step3_reformat_prompt_cnt_list[idx]] = ["mainObject"]
                predicted_list[step3_reformat_prompt_cnt_list[idx]] = reformatted
        assert all(is_final_list), "Not all objects were successfully processed."
        return all_objects_list

    def batch_extract_nouns(self, img_list, sentences):
        if self.ext_type == "tech":
            input_sequences = sentences
            input_ids = self.tokenizer(
                input_sequences, return_tensors="pt", truncation=True, padding=True
            ).input_ids
            output = self.model.generate(input_ids, no_repeat_ngram_size=3, num_beams=4)
            predicted = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            key_words_list = []
            for idx, predicted_sentence in enumerate(predicted):
                key_words = [
                    word.strip()
                    for word in predicted_sentence.split(",")
                    if word.strip()
                ]
                # Check for pronouns and add a placeholder if found
                words_in_sentence = sentences[idx].lower().split()
                if any(pronoun in words_in_sentence for pronoun in self.pronoun):
                    key_words.append("mainObject")
                if any(pronoun in words_in_sentence for pronoun in self.person_pronoun):
                    key_words.append("mainPerson")
                if len(key_words) > self.top_k:
                    key_words = key_words[: self.top_k]
                key_words_list.append(key_words)
            return key_words_list
        elif self.ext_type == "vllm" or self.ext_type == "step_vllm":
            # key_words_list = []
            # for img, sentence in zip(img_list, sentences):
            #     key_words = self.extract_nouns(img, sentence)
            #     key_words_list.append(key_words)
            # return key_words_list
            return self.extract_keywords_step(img_list, sentences)
            
        else:
            raise ValueError("Invalid extraction type. Choose 'tech' or 'vllm'.")


# Example usage
if __name__ == "__main__":
    extractor = WordExtractor()
    sentence = "what is the cost of this scooter?"
    nouns = extractor.extract_nouns(sentence)
    print("Extracted nouns:", nouns)
    sentences = ["what is the cost of this scooter?", "how to use this product?"]
    nouns_list = extractor.batch_extract_nouns(sentences)
    print("Extracted nouns list:", nouns_list)
