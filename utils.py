# The vanilla Qwen3VL prompt also adopted by ScreenSpot-Pro-GUI-Grounding repo for Qwen3-VL model evaluation
from abc import abstractmethod
import copy
import json
import os
from typing import List, Tuple
from torch.utils.data import Dataset
import glob
from PIL import Image
import base64

import logging
import hashlib

logger = logging.getLogger(__name__)

QWEN3VL_GROUNDING_SYSTEM_MESSAGE_TEXT = """
You are a helpful assistant. The user will give you an instruction, and you MUST left click on the corresponding UI element via tool call. If you are not sure about where to click, guess a most likely one.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. \n* You can only use the left_click action to interact with the computer.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `left_click`: Click the left mouse button with coordinate (x, y).", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`.", "type": "array"}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
""".strip()

QWEN3VL_GROUNDING_W_REFUSAL_SYSTEM_MESSAGE_TEXT = """
You are a helpful assistant. The user will give you an instruction, and you MUST left click on the corresponding UI element via tool call. If you are not sure about where to click, guess a most likely one.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. \n* You can only use the left_click action to interact with the computer or refusal if the task is infeasible.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `left_click`: Click the left mouse button with coordinate (x, y).\n* `refusal`: If the task is infeasible (e.g., the task is not related to the image), use the refusal action.", "enum": ["left_click", "refusal"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`.", "type": "array"}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
""".strip()

class BaseLazyDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 annotation_files_glob_pattern: str, # e.g., "annotations/element_grounding/*.json"
                 system_message_text: str,
                 image_dir_relative_path: str = None, # e.g., "images" or "images/element_grounding"
                 sampling_rate: float = 1.0, 
                 message_format: str = 'hf', # 'hf' or 'openai', differ in how image item is represented in the messages
                 image_format: str = 'pil_object', # 'pil_object', 'path', or 'base64', differ in how the image is represented in the messages
        ): # sampling_rate is for debugging, set to a value in (0, 1] to only use a portion of the dataset
        self.data_dir = data_dir
        self.system_message_text = system_message_text
        # glob files in annotation_files_glob_pattern
        self.annotation_files = glob.glob(os.path.join(data_dir, annotation_files_glob_pattern))
        if not self.annotation_files:
            raise ValueError(f"No annotation files found with pattern: {annotation_files_glob_pattern} in directory: {data_dir}")
        self.annotation_files.sort() # sort the annotation files to make sure the order is deterministic
        logger.info(f"Found {len(self.annotation_files)} annotation files with pattern: {annotation_files_glob_pattern} in directory: {data_dir}")
        self.images_dir = os.path.join(data_dir, image_dir_relative_path)
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory not found: {self.images_dir}")
        self.images_zip_file = None
    
        self.examples = None
        self.sampling_rate = sampling_rate
        self.message_format = message_format
        assert self.message_format in ['hf', 'openai'], f"Unsupported format: {self.message_format}"
        self.image_format = image_format
        assert self.image_format in ['pil_object', 'path', 'base64'], f"Unsupported image format: {self.image_format}"
        if self.message_format == 'openai' and not self.image_format == 'base64':
            # for vllm serve mode, the accepted message format is openai, and for image, best in base64
            raise ValueError(f"For openai message format, the only accepted image format is base64. Got image_format={self.image_format}")
        if self.message_format == 'hf' and self.image_format == 'base64':
            # for hf local inference, no need to convert to base64 and decode which is purely wasting of compute
            raise ValueError(f"For hf message format, the accepted image formats are pil_object and path. Got image_format={self.image_format}")

    def _lazy_init(self):
        if self.examples is not None:
            return
        self.examples = []
        for json_file in self.annotation_files:
            rel_path = os.path.relpath(json_file, self.data_dir)
            with open(json_file, 'r') as f:
                data = json.load(f)
                for example in data:
                    example['source_annotation_file'] = rel_path # add the source annotation file path, screenspot-v2 will need this info
                    self.examples.append(example)

        logger.info(f"Loaded {len(self.examples)} examples from annotation files.")
        if 0 < self.sampling_rate < 1.0:
            import random
            random.seed(42)
            random.shuffle(self.examples)
            n_examples_to_sample = int(len(self.examples) * self.sampling_rate)
            self.examples = self.examples[:n_examples_to_sample]
            logger.info(f"Sampled {n_examples_to_sample} examples with sampling_rate={self.sampling_rate}")
        
    def __len__(self):
        if self.examples is None:
            self._lazy_init()
        return len(self.examples)
    
    # to be implemented by subclass to prepare the messages and other fields in the example as needed for evaluation
    # typically, need to add fields "image_path", "instruction", "image_size" if they do not exist
    def _prepare_example(self, example):
        if 'id' not in example:
            json_str = json.dumps(example)
            # cal md5 hash of the json_str as the id
            example_id = hashlib.md5(json_str.encode('utf-8')).hexdigest()
            example['id'] = example_id
    
    def __getitem__(self, idx):
        if self.examples is None:
            self._lazy_init()
        
        example = self.examples[idx]
        example = copy.deepcopy(example) # to avoid modifying the original example in self.examples
        self._prepare_example(example)
        if 'id' not in example:
            raise ValueError(f"Example does not have an 'id' field, which is required for evaluation. Please make sure the annotation files have an 'id' field for each example. Example content: {example}")
        image_file, instruction, image_size = example["image_path"], example["instruction"], example["image_size"]
        image_file = os.path.join(self.images_dir, image_file)
        if not os.path.exists(image_file):
            raise ValueError(f"Image file not found: {image_file}")
        
        if self.image_format == 'pil_object':
            image_object = Image.open(image_file).convert('RGB')        
            if self.message_format == 'hf':
                image_message_item = {"type": "image", "image": image_object}
            else:
                raise ValueError(f"PIL object only works for hf message format. Current message format: {self.message_format}")
        
        elif self.image_format == 'path':
            if self.message_format == 'hf':
                image_message_item = {"type": "image", "path": image_file}
            else:
                raise ValueError(f"Path format only works for hf message format. Current message format: {self.message_format}")
        
        elif self.image_format == 'base64':
            with open(image_file, 'rb') as img_file:
                image_bytes = img_file.read()
            
            base64_str = base64.b64encode(image_bytes).decode('utf-8')
            image_url = f"data:image/png;base64,{base64_str}"
            if self.message_format == 'openai':
                image_message_item = {"type": "image_url", "image_url": {"url": image_url}}
            else:
                raise ValueError(f"Base64 format only works for openai message format. Current message format: {self.message_format}")
        else:
            raise ValueError(f"Unsupported image format: {self.image_format}")
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_message_text}]},
            {
                "role": "user", 
                "content": [
                    image_message_item,
                    {"type": "text", "text": instruction}, 
                ]
            }
        ]
        example['messages'] = messages
        example['image_size'] = image_size
        return example

# extract as pixel coordinates
def extract_coordinates(assistant_message_text: str, image_width: int, image_height: int) -> Tuple[int, int]:
    start = assistant_message_text.index("<tool_call>") + len("<tool_call>")
    end = assistant_message_text.index("</tool_call>")
    json_str = assistant_message_text[start:end].strip()
    json_obj = json.loads(json_str)
    arguments = json_obj["arguments"]
    if arguments["action"] == "refusal": # osworld-G need this
        # return a negative coordinate to indicate refusal
        return (-1, -1)
    
    coordinate = arguments["coordinate"]
    x, y = coordinate
    if x < 0 or y < 0:
        return (-1, -1)
    
    x = round(image_width * x / 1000)
    y = round(image_height * y / 1000)
    return (x, y)

def avg_points(points: List[Tuple[int, int]]):
    if not points:
        return None
    if len(points) == 1:
        return points[0]
    x_avg = round(sum(p[0] for p in points) / len(points))
    y_avg = round(sum(p[1] for p in points) / len(points))
    return (x_avg, y_avg)


def get_predict(generated_texts: List[str], image_width: int, image_height: int, aggregation: str = 'avg') -> Tuple[int, int]:
    if not generated_texts:
        return None
    
    points = []
    for text in generated_texts:
        try:
            point = extract_coordinates(text, image_width, image_height)
            points.append(point)
        except Exception as e:
            logger.warning(f"Error extracting coordinates from generated text: {text}. Error: {e}")
    if not points:
        return None
    if aggregation == 'avg':
        point = avg_points(points)
        return point
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")


def is_point_in_rectangle(point, rect):
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]

# copy from https://github.com/xlang-ai/OSWorld-G/blob/main/evaluation/eval.py
def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon) // 2
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i * 2], polygon[i * 2 + 1]
        xj, yj = polygon[j * 2], polygon[j * 2 + 1]

        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i

    return inside

