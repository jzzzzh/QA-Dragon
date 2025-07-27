from PIL import Image
from io import BytesIO
import os
import time
from typing import Any, List
from urllib.request import urlopen


def download_image(url, timeout=60):
    with urlopen(url, timeout=timeout) as response:
        image_data = response.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image


def repair_images(
        images: List[Image.Image | None], 
        image_urls: List[str | None], 
        batch_offset: int, 
        config: Any,
        retry_limit: int = 3,
        retry_timeout: int = 3
    ) -> List[Image.Image]:
        img_save_dir = config.dataset.img_dir
        os.makedirs(img_save_dir, exist_ok=True)
        
        err_cnt = 0
        concatenated_images = []
        for idx, (img, url) in enumerate(zip(images, image_urls)):
            if img is not None:
                concatenated_images.append(img)
            elif url is not None:
                img_local_path = os.path.join(img_save_dir, f"{batch_offset + idx}.jpg")
                if os.path.exists(img_local_path):
                    image = Image.open(img_local_path).convert("RGB")
                    concatenated_images.append(image)
                else:
                    image = None
                    for i in range(retry_limit):
                        try:
                            print(f"Downloading image from {url} (attempt {i + 1})")
                            image = download_image(url)
                            image.save(img_local_path)
                            break
                        except Exception as e:
                            time.sleep(retry_timeout)
                    
                    if image is not None:
                        concatenated_images.append(image)
                    else:
                        print(f"Failed to fetch the {err_cnt}th image from {url}")
                        err_cnt += 1
                        image = Image.new("RGB", (config.basic.max_img_size, config.basic.max_img_size), (0, 0, 0))
                        concatenated_images.append(image)
        return concatenated_images
    

def resize_image(image, max_size=512, keep_aspect_ratio=True, enlarge=False):
    """
    Resize an image to fit within a specified maximum size while optionally maintaining its aspect ratio.
    Parameters:
        image (PIL.Image.Image): The image to be resized. Must be a valid PIL Image object.
        max_size (int, optional): The maximum size (in pixels) for the width or height of the resized image. 
                                  Defaults to 512.
        keep_aspect_ratio (bool, optional): Whether to maintain the aspect ratio of the original image. 
                                            Defaults to True.
        enlarge (bool, optional): Whether to allow enlarging the image if its dimensions are smaller than 
                                  the specified max_size. Defaults to False.
    Returns:
        PIL.Image.Image: The resized image.
    Raises:
        ValueError: If the input image is None.
    Function Note:
        - If the image's width or height exceeds `max_size`, or if `enlarge` is set to True, the image will 
          be resized.
        - When `keep_aspect_ratio` is True, the resizing logic calculates the new dimensions based on the 
          original aspect ratio to avoid distortion.
        - If `keep_aspect_ratio` is False, the image will be resized to a square with dimensions 
          (`max_size`, `max_size`).
    """
    
    if image is None:
        raise ValueError("Image is not loaded. Please load an image before resizing.")
    if image.width > max_size or image.height > max_size or enlarge:
        aspect_ratio = image.width / image.height
        if aspect_ratio > 1:
            new_width = max_size
            if keep_aspect_ratio:
                new_height = int(max_size / aspect_ratio)
            else:
                new_height = max_size
        else:
            new_height = max_size
            if keep_aspect_ratio:
                new_width = int(max_size * aspect_ratio)
            else:
                new_width = max_size
        image = image.resize((new_width, new_height))
    return image


def extract_valid_samples(batch_data: list, condition_func=lambda x: x != "") -> tuple[list, list]:
    """
    Extract valid samples from a batch based on a condition function.
    
    Args:
        batch_data: List of data samples
        condition_func: Function that returns True for valid samples
        
    Returns:
        Tuple of (valid_indices, valid_samples)
    """
    valid_pairs = [(i, x) for i, x in enumerate(batch_data) if condition_func(x)]
    return zip(*valid_pairs) if valid_pairs else ([], [])