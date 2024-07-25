import os
import torch
import torch.distributed as dist
import numpy as np
import random

from core.utils.utils import make_dir, load_json, save_json

### DDP utils


def sync_nodes(is_ddp):
    if is_ddp:
        dist.barrier()
    else:
        pass


def cleanup(is_ddp):
    if is_ddp:
        dist.destroy_process_group()
    else:
        pass


def get_node_count():
    world_size = os.environ.get("WORLD_SIZE", default=1)
    return int(world_size)


def set_global_seed(seed, rank):
    if seed != -1:
        torch.manual_seed(seed + rank)
        torch.mps.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def setup_DDP(seed=-1):
    # torch.backends.mps.enabled = True
    world_size = get_node_count()
    if world_size > 1:
        dist.init_process_group("gloo")
        rank = dist.get_rank()
        device = torch.device("mps")


if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
