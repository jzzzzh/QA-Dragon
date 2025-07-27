import sys
from typing import List

import torch
from PIL import Image

sys.path.append('.')
from agents.lavis.models import load_model_and_preprocess


class BLIP2Router():
    META_DATA = {
        "complexity_classes": {
            "easy": 0,
            "hard": 1
        },
        "domain_classes": {
            "living beings": 0,
            "food": 1,
            "landmarks": 2,
            "vehicle": 3,
            "text or chart reasoning": 4,
            "shopping product info": 5,
            "other": 6
        }
    }

    def __init__(self, model_name="blip2_router", model_type="router", dtype=torch.float16):
        self.dtype = dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name=model_name, model_type=model_type, is_eval=True, device=self.device
        )

        self.blip_model.to(self.dtype)

        self.label2complexity = {
            val: key for key, val in self.META_DATA["complexity_classes"].items()
        }
        self.label2domain = {
            val: key for key, val in self.META_DATA["domain_classes"].items()
        }

    def __call__(self, images: List[Image.Image], queries: List[str], router_type: str="domain") -> List[str]:
        images = [self.vis_processors["eval"](image) for image in images]
        images = torch.stack(images, dim=0).to(self.device, dtype=self.dtype)  # (B, 3, 224, 224)
        queries = [self.txt_processors["eval"](query) for query in queries]

        query_input = {"image": images, "text": queries}

        complexity_logits, domain_logits = self.blip_model.classify(query_input)

        if router_type == "complexity":
            complexity_labels = complexity_logits.argmax(dim=-1).tolist()
            return [self.label2complexity[label] for label in complexity_labels]
        elif router_type == "domain":
            domain_labels = domain_logits.argmax(dim=-1).tolist()
            return [self.label2domain[label] for label in domain_labels]
        elif router_type == "both":
            complexity_labels = complexity_logits.argmax(dim=-1).tolist()
            domain_labels = domain_logits.argmax(dim=-1).tolist()
            return [
                {
                    "complexity": self.label2complexity[label],
                    "domain": self.label2domain[domain_label]
                }
                for label, domain_label in zip(complexity_labels, domain_labels)
            ]
        else:
            raise ValueError("router_type must be either 'complexity' or 'domain'.")


if __name__ == "__main__":
    # Example usage
    question = "What is the name of this store?"
    image = Image.open("./docs/dataset_info/multiqa.jpg")
    
    router = BLIP2Router()
    # To classify the complexity of the question
    result = router(images=[image], queries=[question], router_type="complexity")
    print(result)

    # To classify the domain of the question
    result = router(images=[image], queries=[question], router_type="domain")
    print(result)

    # To classify both complexity and domain
    result = router(images=[image], queries=[question], router_type="both")
    print(result)

