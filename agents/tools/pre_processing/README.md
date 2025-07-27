# Preprocessing Documentation

### Design

The preprocessing pipeline is divided into the following components:

1. **WordExtractor (`we`)**: Extracts key words or phrases from the input query to guide the preprocessing.
    If no keywords are extracted or no images are segmented, the system automatically adds the keyword `mainobj` and retries the segmentation process. Additionally, if the query contains pronouns like "this" or "that" along with a question mark (`?`), the keyword `mainobj` is also added. For queries involving person-related pronouns, the keyword `mainperson` is added by default.
2. **OCRProcessor (`ocr`)**: Performs Optical Character Recognition (OCR) on the input image to extract textual information.
    The OCRProcessor provides two methods for text extraction:

    - **EasyOCR**: A lightweight OCR library that supports multiple languages and is easy to use.
    - **HuggingFace OCR**: A more advanced OCR solution leveraging HuggingFace's transformer models for improved accuracy and contextual understanding.

3. **ImageSegmenter (`seg`)**: Segments the input image into meaningful regions for further analysis.
    **Segmentation Strategy**: The segmentation process combines direct segmentation and refined segmentation using sliders. Initially, direct segmentation is performed. If the segmented object's size exceeds 50% of the image, the corresponding keyword skips the slider-based segmentation. For all other keywords, a detailed slider-based segmentation is applied to ensure precision.

4. **ImageCaption (`cap`)**: Generates descriptive captions for the input image to provide contextual information.
    - **HuggingFace Caption**: Utilizes BLIP (Bootstrapped Language-Image Pretraining) and BERT models to generate detailed and contextually rich captions for the input image.
    - **vLLM Caption**: Leverages vLLM (Vision-Language Large Models) to produce captions with a focus on understanding complex visual and textual relationships.



## Preprocessing Code Example

```python
from PIL import Image
from omegaconf import OmegaConf
from preprocessing import WordExtractor, OCRProcessor, ImageSegmenter, ImageCaption, ImagePreprocessor

image_path = "docs/dataset_info/singleqa.jpg"
query = "what is the cost of this scooter?"
we = WordExtractor()
ocr = OCRProcessor()
seg = ImageSegmenter()
cap = ImageCaption()
config = OmegaConf.load("configs/basic.yaml").preprocessing
image = Image.open(image_path)
preprocessor = ImagePreprocessor(we=we, ocr=ocr, seg=seg, cap=cap, config=config)
result, keywords_list = preprocessor.preprocess_image(image, query)
```


## Preprocessing Result
All formats are the same, even if disabled, with empty output if disabled, and will always output the original image with the keyword "query."

```python
{
    'scooter cost': [
        {
            'img': <PIL.Image.Image image mode=RGB size=115x82 at 0x7BA0345C3970>,
            'ocr_result': '',
            'caption': 'a picture of a red motorcycle with a black seat and kickstand is parked on the street next to a blue car, with a white line marking the edge of the parking space'
        }
    ],
    'origin_image': [
        {
            'img': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=277x369 at 0x7BA0344E6E00>,
            'ocr_result': '',
            'caption': 'a picture of a man in a black coat stands next to a red motorcycle parked in front of a tall building, with a leafless tree in the background and a clear blue sky above'
        }
    ]
}


['Scooter', 'Cost']
```


When all preprocessing components are disabled, the output will only include the original image with no additional processing applied. This ensures that the system can still return the unaltered input for reference.


```python
{
    'origin_image': [
        {
            'img': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=277x369 at 0x7BA0344E6E00>,
            'ocr_result': '',
            'caption': ''
        }
    ]
}

['what is the cost of this scooter?']
```
