import torch
from PIL import Image
from typing import List
import sys
sys.path.append(".")
from agents.lavis.models import load_model_and_preprocess


class Blip2ITM():
    def __init__(self, model_name="blip2_image_text_matching", model_type="pretrain", dtype=torch.float16):
        self.dtype = dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        blip_model, vis_processors, txt_processors = load_model_and_preprocess(
            name=model_name, model_type=model_type, is_eval=True, device=self.device
        )

        self.txt_processor = txt_processors["eval"]
        self.vis_processor = vis_processors["eval"]

        self.blip_model = blip_model
        self.blip_model.to(self.dtype)

    def match(self, 
        questions: List[str], 
        images: List[Image.Image]
    ) -> List[float]:
        """
        Match questions with images and return scores.

        Args:
            questions (List[str]): List of questions.
            images (List[Image.Image]): List of images.
        
        Returns:
            List[float]: List of matching scores.
        """
        assert len(questions) == len(images), "Questions and images must have the same length."

        images = [self.vis_processor(image) for image in images]
        images = torch.stack(images, dim=0).to(self.device, dtype=self.dtype)  # (B, 3, 224, 224)
        questions = [self.txt_processor(question) for question in questions]

        query_input = {"image": images, "text_input": questions}

        itm_output = self.blip_model(query_input, match_head="itm")
        itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
        itm_scores = itm_scores[:, 1].tolist()

        return itm_scores
    
    def match_multiple_images(
        self, 
        questions: List[str], 
        images: List[List[Image.Image]],
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Match questions with multiple images and return scores.

        Args:
            questions (List[str]): List of questions.
            images (List[List[Image.Image]]): List of lists of images.
            batch_size (int): Batch size for processing images.
        
        Returns:
            List[List[float]]: List of lists of matching scores.
        """
        concat_questions, concat_images = [], []
        for question, _images in zip(questions, images, strict=True):
            concat_questions.extend([question] * len(_images))
            concat_images.extend(_images)
        
        concat_scores = []
        for i in range(0, len(concat_questions), batch_size):
            batch_questions = concat_questions[i:i + batch_size]
            batch_images = concat_images[i:i + batch_size]
            scores = self.match(batch_questions, batch_images)
            concat_scores.extend(scores)

        scores = []
        cursor = 0
        for _images in images:
            scores.append(concat_scores[cursor:cursor + len(_images)])
            cursor += len(_images)
        
        return scores
    
    def __call__(self, questions: List[str], images: List[Image.Image | List[Image.Image]], Threshold = 0.1) -> List[Image.Image | List[Image.Image]]:
        """
        Call the match method directly for convenience.

        Args:
            questions (List[str]): List of questions.
            images (List[Image.Image | List[Image.Image]]): List of images.
            Threshold (float): Threshold for matching scores.
        Returns:
            List[Image.Image | List[Image.Image]]: List of matching scores.
        """
        filtered_img_list = []
        assert len(questions) == len(images), "Questions and images must have the same length."
        if any(isinstance(img, list) for img in images):
            # [[Image.Image, Image.Image, ...], [Image.Image, None], ...]
            valid_images = [[img for img in _ if isinstance(img, Image.Image)] for _ in images]
            scores = self.match_multiple_images(questions, valid_images)
            
            for score_list, _images in zip(scores, images):
                filtered_img_list.append([
                    img if isinstance(img, Image.Image) and score > Threshold else None
                    for img, score in zip(_images, score_list)
                ])
            
            return filtered_img_list
        
        elif any(isinstance(img, Image.Image) for img in images):
            # [None, Image.Image, None, ...]
            valid_pairs = [(q, img) for q, img in zip(questions, images) if isinstance(img, Image.Image)]
            if not valid_pairs:
                return [None] * len(images)
            valid_questions, valid_images = zip(*valid_pairs)
            valid_scores = self.match(list(valid_questions), list(valid_images))
            result = [None] * len(images)
            for idx, (img, score) in enumerate(zip(images, valid_scores)):
                if isinstance(img, Image.Image) and score > Threshold:
                    result[idx] = img
            return result
        else:
            if all(img is None for img in images):
                result = [None] * len(images)
                return result
            else:
                raise ValueError("Images must be either a list of Image objects or a list of lists of Image objects.")
        



if __name__ == "__main__":
    # Example usage
    model = Blip2ITM()
    questions = ["What is the name of this store?",
                 "What is the capital city of this painter's country?"]
    
    images = [Image.open("./docs/dataset_info/multiqa.jpg"), 
              Image.open("./docs/dataset_info/multiqa.jpg")]
    
    scores = model.match(questions, images)
    print(scores)

    multi_images = [
        [Image.open("./docs/dataset_info/multiqa.jpg"), Image.open("./docs/dataset_info/multiqa.jpg")],
        [Image.open("./docs/dataset_info/multiqa.jpg")]
    ]

    multi_scores = model.match_multiple_images(questions, multi_images)
    print(multi_scores)
    filtered_images = model(questions, multi_images, Threshold=0.1)
    print(filtered_images)
    filtered_images = model(questions, images, Threshold=0.1)
    print(filtered_images)
