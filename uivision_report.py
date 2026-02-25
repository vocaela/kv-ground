# NOTE: This file is adapted from https://github.com/ZJUSCL/MVP/blob/master/mvp_osworldg_qwen3vl.py

import logging

from utils import is_point_in_rectangle

logger = logging.getLogger(__name__)


def evaluate(results):
    n_correct = 0
    n_wrong_format = 0
    for example in results:
        pred = example['pred']
        bbox = example['bbox'] # uivision bbox is standard x1,y1,x2,y2 format
        if pred is None:
            example['correctness'] = "wrong_format"
            n_wrong_format += 1
            continue
        if is_point_in_rectangle(pred, bbox):
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

    
