# NOTE: This file is adapted from https://github.com/ZJUSCL/MVP/blob/master/mvp_osworldg_qwen3vl.py

import logging
from typing import List, Tuple

from utils import is_point_in_polygon, is_point_in_rectangle

logger = logging.getLogger(__name__)

def judge_correctness(point: Tuple[int, int]|List[int], boxes_coordinates: Tuple[int, ...]|List[int], boxes_type: str) -> bool:
    if boxes_type == "bbox":
        return is_point_in_rectangle(point, boxes_coordinates)
    elif boxes_type == "polygon":
        return is_point_in_polygon(point, boxes_coordinates)
    elif boxes_type == "refusal":
        # all the center point should be negative
        return all(p < 0 for p in point)
    else:
        raise ValueError(f"Unsupported boxes type: {boxes_type}")

def evaluate(results):
    n_correct = 0
    n_wrong_format = 0
    for example in results:
        pred = example['pred']
        box_coordinates = example['box_coordinates']
        boxes_type = example['box_type']
        if boxes_type == "bbox":
            assert len(box_coordinates) == 4, f"bbox box_coordinates should have 4 values (x, y, w, h), but got {len(box_coordinates)} values: {box_coordinates}"
            # note: when 'bbox' cases, osworld-g uses (x, y, w, h) format for box_coordinates, need to convert to (x1, y1, x2, y2) format for the judge_correctness function. This only needs to be done for 'bbox' cases, not for 'polygon' cases.
            box_coordinates[2:] = [box_coordinates[0] + box_coordinates[2], box_coordinates[1] + box_coordinates[3]] # convert (x, y, w, h) to (x1, y1, x2, y2)
        
        if pred is None:
            example['correctness'] = "wrong_format"
            n_wrong_format += 1
            continue
        if judge_correctness(pred, box_coordinates, boxes_type):
            example['correctness'] = "correct"
            n_correct += 1
        else:
            example['correctness'] = "wrong"

    accuracy = n_correct / len(results)
    logger.info(f"Evaluation results: {n_correct}/{len(results)} correct, accuracy: {accuracy:.4f}")
    report = {
        "metrics": {
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_wrong_format": n_wrong_format,
            "n_total": len(results),
        },
        "details": results,
    }
    return report

    
