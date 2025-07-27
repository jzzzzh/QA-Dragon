import logging
import torch
from typing import List, Dict
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import sys
sys.path.append(".")


class ImageSegmenter:
    def __init__(self, config):
        model_id = "IDEA-Research/grounding-dino-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            self.device
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.config = config
        self.if_rerank = self.config.tools.segmentation.config.is_rank
        self.seg_type = self.config.tools.segmentation.config.seg_type
        self.padding = self.config.tools.segmentation.config.padding
        

    def enhance_image(self, image):
        enhanced_image = image
        return enhanced_image

    def rerank_image(
        self, images: Dict[str, List[Dict[str, Image.Image]]]
    ) -> Dict[str, List[Dict[str, Image.Image]]]:
        reranked_images = {}
        for label, img_list in images.items():
            reranked_images[label] = sorted(
                img_list,
                key=lambda x: x["img"].size[0] * x["img"].size[1],
                reverse=True,
            )

        return reranked_images
    

    def simple_segment_image(self, source, names=["person"]):
        print("===" * 10 + "Warning: segment_image is deprecated(2025/06/12). Please use batch_segment_images instead." + "===" * 10)
        print("===" * 10 + "batch_segment_images currently lacks edge cropping, screen area ratio filtering, and other discard functionalities." + "===" * 10)
        try:
            if isinstance(source, str):
                image = Image.open(source).convert("RGB")
            elif isinstance(source, Image.Image):
                image = source.convert("RGB")
            else:
                print("===" * 10 + "Error: Unsupported image type" + "===" * 10)
                print(source)
                print("===" * 20)
                image = source

            if names is dict:
                print(
                    "===" * 10
                    + f"Warning: Unsupported names type {type(names)}"
                    + "===" * 10
                )
                print(names)
                print("===" * 20)
                # names = list(names.keys())
                names = [f"{k}:{v}" for k, v in names.items()]
            elif names is list:
                for i in range(len(names)):
                    if isinstance(names[i], dict):
                        names[i] = (
                            f"{list(names[i].keys())[0]}:{list(names[i].values())[0]}"
                        )
                    elif isinstance(names[i], str):
                        pass
            else:
                names = ["mainObject"]

            if names is None or len(names) == 0:
                names = ["mainObject"]

            text = ". ".join(names) + "."
            segmented_images = {}
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]],
            )
            for result in results:
                boxes = result["boxes"].cpu().numpy()
                labels = result["text_labels"]
                scores = result["scores"].cpu().numpy()

                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box
                    x1 = max(0, x1 - image.size[0] * self.padding)
                    y1 = max(0, y1 - image.size[1] * self.padding)
                    x2 = min(image.size[0], x2 + image.size[0] * self.padding)
                    y2 = min(image.size[1], y2 + image.size[1] *self.padding)
                    cropped_region = image.crop((x1, y1, x2, y2))
                    if segmented_images.get(label) is None:
                        segmented_images[label] = []
                    segmented_images[label].append(
                        {"img": cropped_region, "score": score}
                    )
            return segmented_images
        except Exception as e:
            print("===" * 10 + "Error: " + str(e) + "===" * 10)
            print(source)
            print("===" * 20)
            return None

    def segment_image(
        self, source, names=["person"], slice_size=1024, overlap_ratio=0.1
    ):
        print("===" * 10 + "Warning: segment_image is deprecated(2025/06/12). Please use batch_segment_images instead." + "===" * 10)
        print("===" * 10 + "batch_segment_images currently lacks edge cropping, screen area ratio filtering, and other discard functionalities." + "===" * 10)
        try:
            if self.seg_type == "sahi":
                if isinstance(source, str):
                    image = Image.open(source).convert("RGB")
                elif isinstance(source, Image.Image):
                    image = source.convert("RGB")
                else:
                    print("===" * 10 + "Error: Unsupported image type" + "===" * 10)
                    print(source)
                    print("===" * 20)
                    image = source
                if names is dict:
                    print(
                        "===" * 10
                        + f"Warning: Unsupported names type {type(names)}"
                        + "===" * 10
                    )
                    print(names)
                    print("===" * 20)
                    # names = list(names.keys())
                    names = [f"{k}:{v}" for k, v in names.items()]
                elif names is list:
                    for i in range(len(names)):
                        if isinstance(names[i], dict):
                            print(
                                "===" * 10
                                + f"Warning: Unsupported names type {type(names)}"
                                + "===" * 10
                            )
                            print(names)
                            print("===" * 20)
                            names[i] = (
                                f"{list(names[i].keys())[0]}:{list(names[i].values())[0]}"
                            )
                        elif isinstance(names[i], str):
                            pass
                else:
                    names = ["mainObject"]

                if names is None or len(names) == 0:
                    names = ["mainObject"]

                text = ". ".join(names) + "."
                segmented_images = {}
                main_label = []
                inputs = self.processor(
                    images=image,
                    text=text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)

                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=0.4,
                    text_threshold=0.3,
                    target_sizes=[image.size[::-1]],
                )
                for result in results:
                    boxes = result["boxes"].cpu().numpy()
                    labels = result["text_labels"]
                    scores = result["scores"].cpu().numpy()

                    for box, label, score in zip(boxes, labels, scores):
                        x1, y1, x2, y2 = box
                        x1 = max(0, x1 - image.size[0] * self.padding)
                        y1 = max(0, y1 - image.size[1] * self.padding)
                        x2 = min(image.size[0], x2 + image.size[0] * self.padding)
                        y2 = min(image.size[1], y2 + image.size[1] * self.padding)
                        cropped_region = image.crop((x1, y1, x2, y2))
                        box_area = (x2 - x1) * (y2 - y1)
                        image_area = image.size[0] * image.size[1]
                        if (
                            box_area / image_area > 0.5
                            or segmented_images.get(label) is not None
                        ):
                            if main_label.count(label) == 0:
                                main_label.append(label)
                            if segmented_images.get(label) is None:
                                segmented_images[label] = []
                            # segmented_images[label].append({'img': cropped_region, 'score': score})
                            segmented_images[label].append({"img": cropped_region})

                width, height = image.size
                step = int(slice_size * (1 - overlap_ratio))
                # Remove main labels from names
                names = [name for name in names if name not in main_label]

                for top in range(0, height, step):
                    for left in range(0, width, step):
                        right = min(left + slice_size, width)
                        bottom = min(top + slice_size, height)
                        patch = image.crop((left, top, right, bottom))
                        # print(type(patch))

                        # Check for specified objects
                        text = ". ".join(names) + "."
                        # print(names)
                        inputs = self.processor(
                            images=patch, text=text, return_tensors="pt"
                        ).to(self.device)
                        with torch.no_grad():
                            outputs = self.model(**inputs)

                        results = self.processor.post_process_grounded_object_detection(
                            outputs,
                            inputs.input_ids,
                            threshold=0.4,
                            text_threshold=0.3,
                            target_sizes=[patch.size[::-1]],
                        )

                        for result in results:
                            boxes = result["boxes"].cpu().numpy()
                            labels = result["text_labels"]
                            scores = result["scores"].cpu().numpy()

                            for box, label, score in zip(boxes, labels, scores):
                                x1, y1, x2, y2 = box
                                x1 = max(0, x1 - image.size[0] * self.padding)
                                y1 = max(0, y1 - image.size[1] * self.padding)
                                x2 = min(image.size[0], x2 + image.size[0] * self.padding)
                                y2 = min(image.size[1], y2 + image.size[1] * self.padding)
                                # Adjust box coordinates to the original image
                                x1 += left
                                y1 += top
                                x2 += left
                                y2 += top
                                cropped_region = image.crop((x1, y1, x2, y2))
                                if segmented_images.get(label) is None:
                                    segmented_images[label] = []
                                # segmented_images[label].append({'img': cropped_region, 'score': score})
                                segmented_images[label].append({"img": cropped_region})
            elif self.seg_type == "simple":
                segmented_images = self.simple_segment_image(source, names)
            else:
                raise ValueError(
                    "Invalid segmentation type. Choose either 'sahi' or 'simple'."
                )
            if self.if_rerank:
                segmented_images = self.rerank_image(segmented_images)
            return segmented_images
        except Exception as e:
            print("===" * 10 + "Error: " + str(e) + "===" * 10)
            print(source)
            print("===" * 20)
            return None

    def _return_black_name_list(self) -> List[str]:
        return ["book", "monitor", "paper", "screen"]
    
    @staticmethod
    def _calculate_aspect_ratio(origin_img, img, threshold=0.05):
        original_width, original_height = origin_img.size
        img_width, img_height = img.size
        width_aspect_ratio = round(original_width / img_width, 2)
        height_aspect_ratio = round(original_height / img_height, 2)
        if abs(width_aspect_ratio - height_aspect_ratio) > threshold:
            logging.warning(
                f"Aspect ratio mismatch between cropped region and original image. Width aspect ratio: {width_aspect_ratio}, Height aspect ratio: {height_aspect_ratio}"
            )
        return width_aspect_ratio, height_aspect_ratio

    def preprocessing_image(
        self, source_list: List[str | Image.Image], names: List[List[str|Dict]]|List[str], origin_source_list: List[Image.Image] | None = None):
        name_list = []
        img_list = []
        seg_img_list = []
        seg_indicate = []
        black_name_list = self._return_black_name_list()
        for index, (source, name) in enumerate(zip(source_list, names)):
            if isinstance(source, str):
                image = Image.open(source).convert("RGB")
            elif isinstance(source, Image.Image):
                image = source.convert("RGB")
            else:
                print("===" * 10 + "Error: Unsupported image type" + "===" * 10)
                print(source)
                print("===" * 20)
                image = source
            img_list.append(image)
            if isinstance(name,dict):
                print(
                    "===" * 10
                    + f"Warning: Unsupported names type {type(name)}"
                    + "===" * 10
                )
                print(name)
                print("===" * 20)
                # names = list(names.keys())
                name = [f"{k}:{v}" for k, v in name.items()]
            elif isinstance(name,list):
                for i in range(len(name)):
                    if isinstance(name[i], dict):
                        name[i] = (
                            f"{list(name[i].keys())[0]}:{list(name[i].values())[0]}"
                        )
                    elif isinstance(name[i], str):
                        pass
            elif isinstance(name, str):
                name = [name]
            else:
                print(type(name))
                name = ["mainObject"]

            if name is None or len(name) == 0:
                print("===" * 10 + "Warning: No name provided" + "===" * 10)
                name = ["mainObject"]
            # Filter out names containing blacklisted words
            name = list(set(name) - set(black_name_list))
            if len(name) == 0:
                # print("===" * 10 + "Warning: All names are blacklisted" + "===" * 10)
                continue
            else:
                text = ". ".join(name) + "."
                seg_indicate.append(index)
                name_list.append(text)
                seg_img_list.append(image)

        if origin_source_list is None:
            origin_source_list = img_list.copy()
        # Group images into batches of 6 if there are more than 6
        grouped_images = []
        grouped_names = []
        for i in range(0, len(seg_img_list), 6):
            grouped_images.append(seg_img_list[i:i + 6])
            grouped_names.append(name_list[i:i + 6])
        inputs = [self.processor(images=tmp_img_list, text=tmp_name_list, return_tensors="pt", padding=True, truncation=True).to(
            self.device
        ) for tmp_img_list, tmp_name_list in zip(grouped_images, grouped_names)]
        with torch.no_grad():
            outputs = [self.model(**input) for input in inputs]

        target_sizes = [[img.size[::-1] for img in tmp_img_list] for tmp_img_list in grouped_images]  # Ensure target_sizes matches batch size
        results_list_list = [self.processor.post_process_grounded_object_detection(
            outputs_item,
            inputs_item.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=target_sizes_idx,
        ) for outputs_item, inputs_item, target_sizes_idx in zip(outputs, inputs, target_sizes)]
        # print(results_list)
        results_list = [result for results in results_list_list for result in results]
        all_result = [{} for _ in range(len(img_list))]
        for results, index in zip(results_list, seg_indicate):
            if not results or len(results["scores"]) == 0 or len(results["boxes"]) == 0 or len(results["text_labels"]) == 0:
                continue
            all_result[index] = results
            
        return img_list, all_result, origin_source_list
        
    def batch_segment_images(
        self, source_list: List[str | Image.Image], names: List[List[str|Dict]]|List[str], origin_source_list: List[Image.Image] | None = None
    ) -> List[Dict[str, List[Dict[str, Image.Image]] | None]]:
        """
        Batch processes and segments images based on provided names and a specified version.
        Args:
            source_list (List[str | Image.Image]): A list of image sources, either file paths (str) or PIL Image objects.
            names (List[List[str | Dict]] | List[str]): A list of names corresponding to the images. Each name can be a string, 
                a list of strings, or a dictionary with key-value pairs.
            version (str, optional): Specifies the segmentation version to use. Defaults to "v2".
                - "v1": Segments all detected objects in the image and returns a dictionary of cropped regions grouped by labels.
                - "v1.1": Similar to "v1", but applies additional filtering based on edge proximity and area ratio.
                - "v2": Segments only the most confident detected object and returns the cropped region directly.
                - "v2.1": Similar to "v2", but applies additional filtering based on edge proximity and area ratio.
                
        Returns:
            List[Dict[str, List[Dict[str, Image.Image]] | None]]: A list of segmented images or cropped regions. 
                - For "v1": Returns a list of dictionaries where each dictionary contains labels as keys and lists of cropped regions with scores as values.
                - For "v1.1": Similar to "v1", but with additional filtering.
                - For "v2": Returns a list of cropped regions corresponding to the most confident detected object for each image.
                - For "v2.1": Similar to "v2", but with additional filtering.
        Notes:
            - Images are grouped into batches of 6 for processing to optimize performance.
            - If no names are provided or unsupported types are encountered, default names are assigned as "mainObject".
            - The function uses a processor and model for object detection and segmentation, and applies padding to the bounding boxes.
            - In "v1", all detected objects are segmented and reranked based on scores.
            - In "v1.1", additional filtering is applied to remove objects near the edges and those that occupy less than 5% of the image area.
            - In "v2", only the most confident detected object is segmented and returned.
            - In "v2.1", similar filtering as in "v1.1" is applied to the most confident detected object.
        Warnings:
            - Unsupported image types or name formats are logged with warnings.
            - If no detection results are found for an image in "v2", the original image is returned.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses. Use SingleImageSegmenter for single image segmentation."
        )





class SingleImageSegmenter(ImageSegmenter):
    """
    A class for segmenting a single image based on provided names.
    Inherits from ImageSegmenter and overrides the segment_image method.
    """

    def __init__(self, config, type="simple", Edge_Threshold = 10, Area_Threshold = 0.05):
        super().__init__(config)
        self.seg_type = type
        self.Edge_Threshold = Edge_Threshold
        self.Area_Threshold = Area_Threshold
        if self.seg_type not in ["hard", "simple"]:
            raise ValueError(
                "Invalid segmentation type. Choose either 'hard' or 'simple'."
            )
        if self.seg_type == "simple":
            print(f"Edge_Threshold: {Edge_Threshold}, Area_Threshold: {Area_Threshold} not used in simple mode of SingleImageSegmenter")
        else:
            print(f"Edge_Threshold: {Edge_Threshold}, Area_Threshold: {Area_Threshold} used in hard mode of SingleImageSegmenter")

    def batch_segment_images(
        self, source_list: List[str | Image.Image], names: List[List[str|Dict]]|List[str], origin_source_list: List[Image.Image] | None = None
    ) -> List[Dict[str, List[Dict[str, Image.Image]] | None]]:
        img_list, all_result, origin_source_list = self.preprocessing_image(source_list, names, origin_source_list)
        segmented_images_list = []
        if self.seg_type == "simple":
            for results, img, origin_img in zip(all_result, img_list, origin_source_list):
                if origin_img is None:
                    origin_img = img
                
                width_aspect_ratio, height_aspect_ratio = self._calculate_aspect_ratio(origin_img, img)
                if not results or len(results["scores"]) == 0 or len(results["boxes"]) == 0 or len(results["text_labels"]) == 0:
                    segmented_images_list.append(origin_img)
                    continue
                box = results["boxes"][0]
                
                x1, y1, x2, y2 = box
                x1 = max(0, x1 * width_aspect_ratio - origin_img.size[0] * self.padding)
                y1 = max(0, y1 * height_aspect_ratio - origin_img.size[1] * self.padding)
                x2 = min(origin_img.size[0], x2 * width_aspect_ratio + origin_img.size[0] * self.padding)
                y2 = min(origin_img.size[1], y2 * height_aspect_ratio + origin_img.size[1] * self.padding)
                cropped_region = origin_img.crop((int(x1), int(y1), int(x2), int(y2)))
                segmented_images_list.append(cropped_region)
        elif self.seg_type == "hard":
            Edge_Threshold = self.Edge_Threshold
            for results, img, origin_img in zip(all_result, img_list, origin_source_list):
                if origin_img is None:
                    origin_img = img

                width_aspect_ratio, height_aspect_ratio = self._calculate_aspect_ratio(origin_img, img)
                if not results or len(results["scores"]) == 0 or len(results["boxes"]) == 0 or len(results["text_labels"]) == 0:
                    segmented_images_list.append(None)
                    continue
                box = results["boxes"][0]
                x1, y1, x2, y2 = box
                x1 *= width_aspect_ratio
                y1 *= height_aspect_ratio
                x2 *= width_aspect_ratio   
                y2 *= height_aspect_ratio
                if x1 <= Edge_Threshold or y1 <= Edge_Threshold or x2 >= origin_img.size[0]-Edge_Threshold or y2 >= origin_img.size[1]-Edge_Threshold:
                    segmented_images_list.append(None)
                    continue
                x1 = max(0, x1 - origin_img.size[0] * self.padding)
                y1 = max(0, y1 - origin_img.size[1] * self.padding)
                x2 = min(origin_img.size[0], x2 + origin_img.size[0] * self.padding)
                y2 = min(origin_img.size[1], y2 + origin_img.size[1] * self.padding)
                box_area = (x2 - x1) * (y2 - y1)
                image_area = origin_img.size[0] * origin_img.size[1]
                if box_area / image_area < self.Area_Threshold:
                    segmented_images_list.append(None)
                    continue
                cropped_region = origin_img.crop((int(x1), int(y1), int(x2), int(y2)))
                segmented_images_list.append(cropped_region)
        return segmented_images_list
        


class MultiImageSegmenter(ImageSegmenter):
    """
    A class for segmenting multiple images based on provided names.
    Inherits from ImageSegmenter and overrides the batch_segment_images method.
    """

    def __init__(self, config, type="simple", seg_img_top_k = 2, Edge_Threshold = 10, Area_Threshold = 0.05, ranking_type="score"):
        super().__init__(config)
        self.seg_type = type
        self.seg_img_top_k = seg_img_top_k
        self.Edge_Threshold = Edge_Threshold
        self.Area_Threshold = Area_Threshold
        self.ranking_type = ranking_type
        if self.ranking_type not in ["score", "size"]:
            raise ValueError(
                "Invalid ranking type. Choose either 'score' or 'size'."
            )
        if self.seg_type not in ["hard", "simple"]:
            raise ValueError(
                "Invalid segmentation type. Choose either 'hard' or 'simple'."
            )
        if self.seg_type == "simple":
            print(f"Edge_Threshold: {Edge_Threshold}, Area_Threshold: {Area_Threshold} not used in simple mode of MultiImageSegmenter")
        else:
            print(f"Edge_Threshold: {Edge_Threshold}, Area_Threshold: {Area_Threshold} used in hard mode of MultiImageSegmenter")

    def batch_segment_images(
        self, source_list: List[str | Image.Image], names: List[List[str|Dict]]|List[str], origin_source_list: List[Image.Image] | None = None
    ) -> List[Dict[str, List[Dict[str, Image.Image]] | None]]:
        img_list, all_result, origin_source_list = self.preprocessing_image(source_list, names, origin_source_list)
        segmented_images_list = []
        if self.seg_type == "simple":
            search_img_list_list = []
            seg_img_top_k = self.seg_img_top_k
            
            for results, img, origin_img in zip(all_result, img_list, origin_source_list):
                flag = True
                all_img_list = []
                all_score_list = []
                all_size_list = []
                # Resize the cropped region to match the original image size while maintaining aspect ratio
                if origin_img is None:
                    origin_img = img
                
                width_aspect_ratio, height_aspect_ratio = self._calculate_aspect_ratio(origin_img, img)
                
                segmented_images = {}
                if not results or len(results["scores"]) == 0 or len(results["boxes"]) == 0 or len(results["text_labels"]) == 0:
                    segmented_images_list.append({"origin_img":[{"img":img, "score": 0.0}]})
                    search_img_list_list.append([img])
                    flag = False
                    continue
                boxes = results["boxes"]
                labels = results["text_labels"]
                scores = results["scores"]
                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box
                    x1 = max(0, x1 * width_aspect_ratio - origin_img.size[0] * self.padding) 
                    y1 = max(0, y1 * height_aspect_ratio - origin_img.size[1] * self.padding) 
                    x2 = min(origin_img.size[0], x2 * width_aspect_ratio + origin_img.size[0] * self.padding) 
                    y2 = min(origin_img.size[1], y2 * height_aspect_ratio + origin_img.size[1] * self.padding)  # Updated to use image.size[1]
                    
                    cropped_region = origin_img.crop((int(x1), int(y1), int(x2), int(y2)))
                    if segmented_images.get(label) is None:
                        segmented_images[label] = []
                    segmented_images[label].append(
                        {"img": cropped_region, "score": score}
                    )
                    all_img_list.append(cropped_region)
                    all_score_list.append(score)
                    all_size_list.append(cropped_region.size[0] * cropped_region.size[1])
                if flag:
                    segmented_images = self.rerank_image(segmented_images)
                    # Select top-k images based on scores
                    if self.ranking_type == "score":
                        top_k_indices = sorted(range(len(all_score_list)), key=lambda i: all_score_list[i], reverse=True)[:seg_img_top_k]
                    elif self.ranking_type == "size":
                        top_k_indices = sorted(range(len(all_size_list)), key=lambda i: all_size_list[i], reverse=True)[:seg_img_top_k]
                    search_img_list = [all_img_list[i] for i in top_k_indices]
                    search_img_list_list.append(search_img_list)
                    segmented_images_list.append(segmented_images)
            return search_img_list_list
            
            
        elif self.seg_type == "hard":
            search_img_list_list = []
            seg_img_top_k = self.seg_img_top_k
            
            Edge_Threshold = self.Edge_Threshold
            for results, img, origin_img in zip(all_result, img_list, origin_source_list):
                flag = True
                all_img_list = []
                all_score_list = []
                all_size_list = []
                search_img_list = []
                if origin_img is None:
                    origin_img = img
                
                width_aspect_ratio, height_aspect_ratio = self._calculate_aspect_ratio(origin_img, img)
                
                segmented_images = {}
                if not results or len(results["scores"]) == 0 or len(results["boxes"]) == 0 or len(results["text_labels"]) == 0:
                    segmented_images_list.append(None)
                    search_img_list_list.append([None])
                    flag = False    
                    continue
                boxes = results["boxes"]
                labels = results["text_labels"]
                scores = results["scores"]
                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box
                    x1 *= width_aspect_ratio
                    y1 *= height_aspect_ratio
                    x2 *= width_aspect_ratio
                    y2 *= height_aspect_ratio
                    if x1 <= Edge_Threshold or y1 <= Edge_Threshold or x2 >= origin_img.size[0]-Edge_Threshold or y2 >= origin_img.size[1]-Edge_Threshold:
                        segmented_images_list.append(None)
                        search_img_list_list.append([None])
                        flag = False
                        break
                    box_area = (x2 - x1) * (y2 - y1)
                    image_area = origin_img.size[0] * origin_img.size[1]
                    if box_area / image_area < self.Area_Threshold:
                        segmented_images_list.append(None)
                        search_img_list_list.append([None])
                        flag = False
                        break
                    x1 = max(0, x1 - origin_img.size[0] * self.padding)
                    y1 = max(0, y1 - origin_img.size[1] * self.padding)  # Updated to use origin_img.size[1]
                    x2 = min(origin_img.size[0], x2 + origin_img.size[0] * self.padding)
                    y2 = min(origin_img.size[1], y2 + origin_img.size[1] * self.padding)
                    cropped_region = origin_img.crop((int(x1), int(y1), int(x2), int(y2)))
                    if segmented_images.get(label) is None:
                        segmented_images[label] = []
                    segmented_images[label].append(
                        {"img": cropped_region, "score": score}
                    )
                    all_img_list.append(cropped_region)
                    all_score_list.append(score)
                    all_size_list.append(cropped_region.size[0] * cropped_region.size[1])
                if flag:
                    segmented_images = self.rerank_image(segmented_images)
                    # Select top-k images based on scores
                    if self.ranking_type == "score":
                        top_k_indices = sorted(range(len(all_score_list)), key=lambda i: all_score_list[i], reverse=True)[:seg_img_top_k]
                    elif self.ranking_type == "size":
                        top_k_indices = sorted(range(len(all_size_list)), key=lambda i: all_size_list[i], reverse=True)[:seg_img_top_k]
                    search_img_list = [all_img_list[i] for i in top_k_indices]
                    search_img_list_list.append(search_img_list)
                    segmented_images_list.append(segmented_images)
            return search_img_list_list
                
    
    
    
# Example usage
if __name__ == "__main__":
    # config_dir = ""
    from omegaconf import OmegaConf
    conf_dir = "configs/hitech.yaml"
    config = OmegaConf.load(conf_dir).preprocessing
    test_source_list = ["docs/dataset_info/singleqa.jpg" for _ in range(9)]
    test_source_list[0] = "viewer/datasets/crag-mm-single-turn-public/validation/000045.jpg"
    test_source_list[1] = "viewer/datasets/crag-mm-single-turn-public/validation/000004.jpg"
    test_names = [["scooter"] for _ in range(9)]
    test_names[0] = ["monitor"]
    test_names[1] = ["car"]
    # v2
    # segmenter = SingleImageSegmenter(config, type="simple", Edge_Threshold=0, Area_Threshold=0.05)
    # v2.1
    # segmenter = SingleImageSegmenter(config, type="hard", Edge_Threshold=0, Area_Threshold=0.05)
    segmenter = MultiImageSegmenter(config, type="simple", seg_img_top_k=2, Edge_Threshold=0, Area_Threshold=0.05, ranking_type="score")
    # segmenter = MultiImageSegmenter(config, type="hard", seg_img_top_k=2, Edge_Threshold=0, Area_Threshold=0.05, ranking_type="score")
    
    seg_img_list = segmenter.batch_segment_images(
        source_list=test_source_list,
        names=test_names
    )
    # Visualize the results
    print("Segmented images:")
    print(seg_img_list)
    print(len(seg_img_list))