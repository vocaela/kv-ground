
import json
import os
import traceback
from typing import Any, Callable, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import shutil
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from torch_dist_utils import SequentialDistributedSampler, pass_through_collate_fn
from utils import get_predict

logger = logging.getLogger(__name__)


# assume already setup distributed before calling this function
def eval(
    rank: int,
    world_size: int,
    local_rank: int,
    output_dir: str,
    report_fn: Callable,
    dataset: Dataset,
    num_workers: int,
    model_dir: str,
    # model kwargs
    dtype: str,
    attn_implementation: str,
    # processor kwargs
    min_pixels: int,
    max_pixels: int,
    # generation kwargs
    num_return_sequences: int = None,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
    max_new_tokens: int = 256,
    sort_key: str = None,
    resume_from_dir: str = None,
    resume_on: bool = False
) -> List[Dict[str, Any]]:
    logger = logging.getLogger(__name__)
    logger.info(f"Distributed setup: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    # log the arguments
    logger.info(f"output_dir={output_dir}")
    logger.info(f"dataset={dataset.__class__.__name__}, num_examples={len(dataset)}")
    logger.info(f"report_fn={report_fn.__name__}")
    logger.info(f"model_dir={model_dir}")
    logger.info(f"model_kwargs: dtype={dtype}, attn_implementation={attn_implementation}")
    logger.info(f"processor_kwargs: min_pixels={min_pixels}, max_pixels={max_pixels}")
    logger.info(f"generation_kwargs: num_return_sequences={num_return_sequences}, temperature={temperature}, top_p={top_p}, top_k={top_k}, max_new_tokens={max_new_tokens}")
    logger.info(f"sort_key={sort_key}")
    logger.info(f"num_workers={num_workers}")
    logger.info(f"resume_from_dir={resume_from_dir}")

    logger.info(f"Starting inference with rank={rank}")

    # This block is for resuming run from interruption in cloud spot vms.
    # For each run, local rank results are saved to a rank-specific file for possible resuming running purpose only
    # And it assumes id field in the example!
    local_rank_result_file_name = f"rank_{rank}_results.jsonl"
    local_rank_result_file = os.path.join(output_dir, local_rank_result_file_name)
    if resume_on and resume_from_dir is not None:
        # copy the local rank result file from the resume_from_dir to output_dir for resuming
        # note that each rank only copy its own local result file to avoid redundant copying and possible
        resume_rank_result_file = os.path.join(resume_from_dir, local_rank_result_file_name)
        shutil.copyfile(resume_rank_result_file, local_rank_result_file)
        logger.info(f"Copied local rank result file from {resume_rank_result_file} to {local_rank_result_file} for resuming purpose")

    local_results = []
    already_run_ids = set()
    if resume_on and os.path.exists(local_rank_result_file):
        logger.warning(f"Local result file {local_rank_result_file} already exists. Will load and skip them...")
        with open(local_rank_result_file, 'r', encoding='utf-8') as fi:
            for line in fi:
                try:
                    res = json.loads(line.strip())
                except Exception as e:
                    logger.warning(f"Error loading line from local result file {local_rank_result_file}: {e}\n line content: {line}")
                    continue

                local_results.append(res)
                already_run_ids.add(res['id'])
        logger.warning(f"Loaded {len(local_results)} results from local result file {local_rank_result_file}. Will skip these examples during inference to avoid redundant computation.")

    # load model in each rank
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        device_map=f"cuda:{local_rank}", # directly map to each rank
        dtype=getattr(torch, dtype),
        attn_implementation=attn_implementation
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(
        model_dir,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    
    if world_size > 1:
        # Distributed sampler
        sampler = SequentialDistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1, # Avoid batch for evaluation to strictly control possible vairants impacting results
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pass_through_collate_fn
    )
    
    if rank == 0:
        pbar = tqdm(total=len(dataloader), desc=f"Processing examples (rank-{rank})")

    generation_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_return_sequences,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    if "temperature" in generation_kwargs and generation_kwargs["temperature"] == 0:
        generation_kwargs["do_sample"] = False
    
    n_skipped = 0
    n_total = 0
    write_mode = 'a' if resume_on else 'w'
    with open(local_rank_result_file, write_mode, encoding='utf-8') as fo:
        for batch in dataloader:
            for example in batch:
                id = example['id']
                n_total += 1
                if id in already_run_ids:
                    n_skipped += 1
                    continue
                try:
                    inputs = processor.apply_chat_template(
                        example['messages'], 
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=True).to(f"cuda:{local_rank}")
                    generated_ids = model.generate(**inputs, **generation_kwargs)
                    input_ids = inputs.input_ids
                    if num_return_sequences is not None and num_return_sequences > 1:
                        # input_ids is one dim now [1, seq_len], need to expand to [num_return_sequences, seq_len] for correct trimming later
                        input_ids = input_ids.expand(num_return_sequences, -1)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
                    ]
                    generated_texts = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
                    )
                except Exception as e:
                    tb = traceback.format_exc()
                    generated_texts = None
                    error = f"Rank-{rank}: error during model generation: {e}\n{tb}\n example: {example}"
                    errors = example.get('errors', [])
                    errors.append(error)
                    example['errors'] = errors
                    logger.warning(error)
                
                result = {k:v for k,v in example.items() if k != "messages"} # free up memory, majorly the image
                result['generated_texts'] = generated_texts
                local_results.append(result)
                # write to local file for possible resuming purpose
                fo.write(json.dumps(result) + '\n')
                fo.flush()
            
            if rank == 0:
                pbar.update(1) # this is one batch, but here batch size always 1 so also the number of examples
                pbar.set_postfix(skipped=n_skipped, total=n_total)

    if rank == 0:
        pbar.close()
    
    # gather results from all ranks
    if world_size > 1:
        all_results = [None] * world_size
        torch.distributed.gather_object(local_results, all_results if rank == 0 else None, dst=0)
        
        if rank == 0:
            results = []
            for res_list in all_results:
                if res_list is not None:
                    results.extend(res_list)
            
            # sort by the specified key to ensure the order
            if sort_key:
                results.sort(key=lambda x: x[sort_key])
        else:
            results = []
    else:
        results = local_results

    # only rank 0 will do evaluation and save the report
    if rank == 0:
        # extract coordinates
        for example in results:
            generated_texts, img_size = example['generated_texts'], example['image_size'] # dataset already standardized the field 'image_size' to be (width, height)
            image_width, image_height = img_size
            try:
                pred = get_predict(generated_texts=generated_texts, image_width=image_width, image_height=image_height)
            except Exception as e:
                tb = traceback.format_exc()
                pred = None
                error = f"Error extracting coordinate: {e}\n{tb}\n example: {example}"
                errors = example.get('errors', [])
                errors.append(error)
                example['errors'] = errors
                logger.warning(error)
            # follow sspro repo schema to re-use the report code
            example['pred'] = pred
            
        # report
        report = report_fn(results)
        report_file = os.path.join(output_dir, "report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        logger.info(f"Saved evaluation report to {report_file}")
        if 'metrics' in report:
            logger.info(f"metrics: {report['metrics']}")
        