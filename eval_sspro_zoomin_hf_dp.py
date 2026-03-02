# This is the 2nd step running of the zoom-in evaluation
# Instead of doing real-time two step prediction in one example, we run in the offline batch mode, first run all the predictions of the 1st step, save the prediction results, then run this code to do the cropping and 2nd step evaluation, this is easy to re-use this eval code base.

import os
import sys
import argparse
import logging

from sspro_report import sspro_zoomin_evaluate as evaluate
from torch_dist_utils import setup_torch_distributed, setup_rank_logger, cleanup_torch_distributed
from hf_dp_eval import eval
from utils import BaseZoomInLazyDataset, QWEN3VL_GROUNDING_SYSTEM_MESSAGE_TEXT as SYSTEM_MESSAGE_TEXT, str2bool

logger = logging.getLogger(__name__)

class SsproZoomInHFLazyDataset(BaseZoomInLazyDataset):
    def __init__(self, 
                data_dir,
                predict_result_file: str,
                sampling_rate: float = 1.0,
                crop_size_ratio: float = 0.5, # the ratio of the cropped sub image size to the original image size, e.g., 0.5 means the cropped sub image will be 1/4 of the original image
                resize_to_origin: bool = True, # whether to resize the cropped sub image back to the original image size
        ):
        super().__init__(
            predict_result_file=predict_result_file,
            crop_size_ratio=crop_size_ratio,
            resize_to_origin=resize_to_origin,
            # base args
            data_dir=data_dir,
            annotation_files_glob_pattern="annotations/*.json",
            image_dir_relative_path="images",
            system_message_text=SYSTEM_MESSAGE_TEXT,
            sampling_rate=sampling_rate,
            message_format='hf',
            image_format='pil_object'
        )

    def _prepare_example(self, example):
        super()._prepare_example(example)
        example['image_path'] = example["img_filename"]
        example["image_size"] = example["img_size"]
        # 'instruction' field already exists

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    # common args
    parser.add_argument('--output_dir', type=str, default=".", help='The output directory to save the evaluation results')
    parser.add_argument('--resume_from_dir', type=str, default=None, help='The directory to resume the evaluation from, if any')
    parser.add_argument('--resume_on', type=str2bool, default=False, help='Whether to turn-on the auto resume mode to automatically resume from the last interrupted run.')

    parser.add_argument('--model_dir', type=str, help='The path to the model checkpoint')
    parser.add_argument('--data_dir', type=str, help='The dir of the original official dataset downloaded from https://huggingface.co/datasets/likaixin/ScreenSpot-Pro/tree/main')
    parser.add_argument('--step1_output_dir', type=str, default=None, help='The output dir of the 1st step evaluation, assume report.json there')
    parser.add_argument('--sampling_rate', type=float, default=1.0, help='The sampling rate for the dataset, set to a value in (0, 1] to only use a portion of the dataset for debugging')
    parser.add_argument('--crop_size_ratio', type=float, default=0.5, help='The ratio of the cropped sub image size to the original image size, e.g., 0.5 means the cropped sub image will be 1/4 of the original image')
    parser.add_argument('--resize_to_origin', type=str2bool, default=True, help='Whether to resize the cropped sub image back to the original image size.')

    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--sort_key', type=str, default=None, help='The key to sort the results before saving the report. Default None without sorting.')
    # model param
    parser.add_argument('--attn_implementation', type=str, default="flash_attention_2", help='Attention implementation, e.g., "flash_attention_2", "sdpa"')
    parser.add_argument('--dtype', type=str, default="bfloat16", help='Data type for model parameters, e.g., "float16", "bfloat16"')
    parser.add_argument('--max_pixels', type=int, default=99999999, help='Maximum number of pixels for input images (after resizing), e.g., 99999999 here means no resizing almost for all cases. Following the setting in ScreenSpot-Pro-GUI-Grounding repo.')
    parser.add_argument('--min_pixels', type=int, default=65536, help='Minimum number of pixels for input images (after resizing), e.g., 65536 for 256x256')
    # inference params
    parser.add_argument('--num_return_sequences', type=int, default=None, help='Number of sequences to return for each input example. If >1, will use the aggregation method specified in get_predict to aggregate the multiple predictions into one coordinate for evaluation. If None, will return a single sequence.')
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for sampling. Set to 0.0 for greedy decoding.')
    parser.add_argument('--top_p', type=float, default=None, help='Top-p sampling parameter. Set to 1.0 to disable.')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling parameter. Set to 0 to disable.')
    
    args = parser.parse_args()

    # setup distributed
    rank, world_size, local_rank = setup_torch_distributed()
    setup_rank_logger()

    logger.info(f"args: {args}")

    if not (args.step1_output_dir and os.path.exists(args.step1_output_dir)):
        raise ValueError("The --step1_output_dir argument must be provided and must exist, which is the output directory of the 1st step evaluation")
    
    # load dataset
    step1_report_file = os.path.join(args.step1_output_dir, "report.json")
    dataset = SsproZoomInHFLazyDataset(args.data_dir, sampling_rate=args.sampling_rate, crop_size_ratio=args.crop_size_ratio, resize_to_origin=args.resize_to_origin, predict_result_file=step1_report_file)
    eval(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        output_dir=args.output_dir,
        report_fn=evaluate,
        dataset=dataset,
        num_workers=args.num_workers,
        model_dir=args.model_dir,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        num_return_sequences=args.num_return_sequences,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        sort_key=args.sort_key,
        resume_from_dir=args.resume_from_dir,
        resume_on=args.resume_on
    )
    
    del dataset
    cleanup_torch_distributed()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Ensure root logger inherits this
    logging.getLogger().setLevel(logging.INFO)
    main()