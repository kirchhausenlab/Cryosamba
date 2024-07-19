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
    world_size = os.environ.get('WORLD_SIZE', default=1)
    return int(world_size)

def set_global_seed(seed, rank):
    if seed!=-1:
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 

def setup_DDP(seed=-1):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    world_size = get_node_count()
    if world_size>1:      
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
    else:
        rank, device = 0, 0

    set_global_seed(seed, rank)
    torch.cuda.set_device(rank)
    
    return world_size, rank, device

### Optimizer utils

def get_optimizer(model, args):
    return torch.optim.AdamW(model.parameters(), args.lr, betas=args.betas, eps=args.epsilon, weight_decay=args.weight_decay)

def get_lr(optimizer):
    if not optimizer.param_groups:
        raise ValueError("Optimizer does not have any parameter groups")
    lr = optimizer.param_groups[0]['lr']
    return lr

class CombinedScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, lr_decay, last_iter=-1):
        self.warmup_steps = warmup_steps
        self.lr_decay = lr_decay
        super().__init__(optimizer, last_epoch=last_iter)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            alpha = float(self._step_count) / self.warmup_steps
            scale_factor = (1 / self.warmup_steps) * (1 - alpha) + alpha
        else:
            scale_factor = self.lr_decay ** (self._step_count - self.warmup_steps)
        
        return [base_lr * scale_factor for base_lr in self.base_lrs]

def get_scheduler(optimizer, warmup_steps, lr_decay, last_iter=-1):
    return CombinedScheduler(optimizer, warmup_steps, lr_decay, last_iter)

# Checkpoint utils

def count_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return "{:,.0f}".format(total_params), "{:,.0f}".format(trainable_params)

def adjust_keys_for_compiled_model(loaded_state_dict):
    """ Prefixes all keys with '_orig_mod.' to fit the structure of a compiled model. """
    return {f'_orig_mod.{k}': v for k, v in loaded_state_dict.items()}

def state_dict_remove_prefix(state_dict, prefix):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if k.startswith(prefix):
            name = name[len(prefix):]
        new_state_dict[name] = v
    return new_state_dict

def state_dict_add_prefix(state_dict, prefix):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if not k.startswith(prefix):
            name = prefix + name
        new_state_dict[name] = v
    return new_state_dict

def fix_state_dict(state_dict, is_ddp, compile):
    state_dict = state_dict_remove_prefix(state_dict, 'module.')
    state_dict = state_dict_remove_prefix(state_dict, '_orig_mod.')
    state_dict = state_dict_remove_prefix(state_dict, 'module._orig_mod.')
    
    if is_ddp and compile:
        state_dict = state_dict_add_prefix(state_dict, 'module._orig_mod.')
    elif is_ddp and not compile:
        state_dict = state_dict_add_prefix(state_dict, 'module.')
    elif not is_ddp and compile:
        state_dict = state_dict_add_prefix(state_dict, '_orig_mod.')

    return state_dict

def save_ckpt(model, optimizer, scheduler, iter, path):
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'iter': iter,
    }
    torch.save(ckpt, path)

def load_ckpt(path, model=None, optimizer=None, scheduler=None, is_ddp=False, compile=False):
    if os.path.exists(path):
        ckpt = torch.load(path)
        if model is not None:
            model.load_state_dict(fix_state_dict(ckpt['model_state_dict'], is_ddp, compile), strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_iter = ckpt['iter']+1
    else:
        start_iter = 0
    return model, optimizer, scheduler, start_iter




