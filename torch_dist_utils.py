
import os
import torch
from torch.utils.data.distributed import DistributedSampler
import logging

logger = logging.getLogger(__name__)

class SequentialDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
        # don't drop as this is for evaluation
        self.drop_last = False

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # get number of examples to process for this rank
        num_samples = len(indices)
        samples_per_rank = num_samples // self.num_replicas
        remainder = num_samples % self.num_replicas

        # assign sample range for the current rank
        start_idx = self.rank * samples_per_rank + min(self.rank, remainder)
        end_idx = start_idx + samples_per_rank + (1 if self.rank < remainder else 0)
        assigned_indices = indices[start_idx:end_idx]

        return iter(assigned_indices)


def pass_through_collate_fn(batch):
    return batch


def setup_torch_distributed():
    # setup distributed env
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)
        return rank, world_size, gpu
    else:
        logger.info("Not using distributed mode")
        return 0, 1, 0


def cleanup_torch_distributed():
    # cleanup distributed env
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# A clean way to avoid "rank0_print" everywhere: only rank 0 process at INFO level logging. Other ranks will only log WARN & ERROR level messages to avoid cluttering the output.
def setup_rank_logger():
    rank = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    
    logger = logging.getLogger()
    level = logging.INFO
    if rank is not None and rank != 0:
        level = logging.WARN
    logger.setLevel(level)
    handler = logging.StreamHandler()
    if rank is not None:
        formatter = logging.Formatter(
            f"[rank {rank}] %(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
    else:
        formatter = logging.Formatter(
            f"%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
    handler.setFormatter(formatter)

    logger.handlers = []   # avoid duplicate handlers
    logger.addHandler(handler)
