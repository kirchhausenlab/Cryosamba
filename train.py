import os, sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
from time import time

import torch
from torch import GradScaler
from torch import autocast

from core.model import get_model, get_loss
from core.dataset import get_dataloader
from core.utils.utils import (
    setup_run,
    load_json,
    logger_info,
    set_writer_train,
    listify,
)
from core.utils.data_utils import get_data
from core.utils.torch_utils import (
    count_model_params,
    setup_DDP,
    sync_nodes,
    cleanup,
    save_ckpt,
    load_ckpt,
    get_lr,
    get_optimizer,
    get_scheduler,
)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Train:
    def __init__(self, args):
        self.run_stamp = time()

        self.world_size, self.rank, self.device = setup_DDP(args.seed)
        self.is_ddp = self.world_size > 1
        cfg = load_json(args.config)

        if self.rank == 0:
            setup_run(cfg, mode="training")
            self.writer = set_writer_train(cfg)
        sync_nodes(self.is_ddp)

        self.log(
            f"Using seed number {args.seed}"
            if args.seed != -1
            else f"No random seed was set"
        )
        self.log(f"# of processes = {self.world_size}")

        # init params

        self.resume_ckpt = os.path.join(cfg.train_dir, "last.pt")
        self.load_ckpt_path = cfg.train.load_ckpt_path
        self.max_frame_gap = cfg.train_data.max_frame_gap
        self.num_iters = cfg.train.num_iters
        self.mixed_precision = cfg.train.mixed_precision
        self.train_dir = cfg.train_dir
        self.print_freq = cfg.train.print_freq
        self.val_freq = cfg.train.val_freq
        self.save_freq = cfg.train.save_freq
        self.compile = cfg.train.compile
        if hasattr(cfg.train, "do_early_stopping"):
            self.do_early_stopping = cfg.train.do_early_stopping
        else:
            self.do_early_stopping = False

        # Init model

        self.model = get_model(
            cfg, self.device, is_ddp=self.is_ddp, compile=self.compile
        )
        self.optimizer = get_optimizer(self.model, cfg.optimizer)
        self.scheduler = get_scheduler(
            self.optimizer, cfg.train.warmup_iters, cfg.optimizer.lr_decay
        )
        self.scaler = GradScaler("cuda", enabled=cfg.train.mixed_precision)
        self.loss_fn = get_loss()

        self.log(f"# of model parameters = {count_model_params(self.model)[1]}")

        # Init dataloaders

        cfg.data_path = listify(cfg.data_path)
        data_list, metadata_list = list(
            zip(*[get_data(path) for path in cfg.data_path])
        )
        self.train_loader = get_dataloader(
            cfg.train_data,
            data_list,
            metadata_list,
            split="train",
            is_ddp=self.is_ddp,
            shuffle=True,
        )
        self.val_loader = get_dataloader(
            cfg.train_data,
            data_list,
            metadata_list,
            split="val",
            is_ddp=False,
            shuffle=False,
        )

        self.log(
            f"# of data samples = {len(self.train_loader)} (train), {len(self.val_loader)} (val)"
        )

        self.early_stopper = EarlyStopper(patience=3, min_delta=0)

    def log(self, message):
        return logger_info(self.rank, message)

    def print_train_loss(self, train_loss, print_time):
        save_ckpt(
            self.model, self.optimizer, self.scheduler, self.iter, self.resume_ckpt
        )

        for key, value in train_loss.items():
            self.writer.add_scalar(f"train_loss/{key}", value, self.iter)
        self.writer.add_scalar("learning_rate/lr", get_lr(self.optimizer), self.iter)
        self.writer.flush()

        message = f"Iter {self.iter}/{self.num_iters} :: train loss: "
        message += ", ".join(
            f"{value:.6f} ({key})" for key, value in train_loss.items()
        )
        message += f", E.T.: {(time()-print_time):.3f}s"
        self.log(message)

    def save_model(self):
        self.zfill = len(str(self.num_iters))
        save_ckpt_path = os.path.join(
            self.train_dir, str(int(self.iter)).zfill(self.zfill) + ".pt"
        )
        save_ckpt(self.model, self.optimizer, self.scheduler, self.iter, save_ckpt_path)

    @torch.inference_mode()
    def validation(self, write=True):
        val_stamp = time()
        self.model.eval()
        val_loss = 0
        for i, imgs in enumerate(self.val_loader):
            imgs = imgs.to(device=self.device)
            imgs = torch.split(imgs, 1, dim=1)
            t = 1
            img0, imgT, img1 = (
                imgs[self.max_frame_gap - t].contiguous(),
                imgs[self.max_frame_gap].contiguous(),
                imgs[self.max_frame_gap + t].contiguous(),
            )

            if self.is_ddp:
                rec = self.model.module.validation(img0, img1)
            else:
                rec = self.model.validation(img0, img1)
            loss = self.loss_fn(rec, imgT)
            val_loss += float(loss.item()) / len(self.val_loader)
        val_time = time() - val_stamp
        self.model.train()

        if write==True:
            self.writer.add_scalar("val_loss/val", val_loss, self.iter)
            self.writer.flush()

        self.log(
            f"Iter {self.iter}/{self.num_iters} :: val loss: {val_loss:.6f}, E.T.: {val_time:.3f}s"
        )

        return val_loss

    def run_training(self):
        # Load checkpoint

        self.model, self.optimizer, self.scheduler, self.iter = load_ckpt(
            self.resume_ckpt,
            self.model,
            self.optimizer,
            self.scheduler,
            is_ddp=self.is_ddp,
            compile=self.compile,
        )
        if self.load_ckpt_path is not None:
            self.model = load_ckpt(
                self.load_ckpt_path,
                self.model,
                is_ddp=self.is_ddp,
                compile=self.compile,
            )[0]

        # Training loop

        if self.rank == 0:
            train_loss = {f"gap {t}": 0 for t in range(1, self.max_frame_gap + 1)}
        epoch = self.iter // len(self.train_loader)
        print_time = time()

        while self.iter <= self.num_iters:
            self.log(f"*** Epoch {epoch} ***")

            self.model.train()
            if self.is_ddp:
                self.train_loader.sampler.set_epoch(epoch)

            for i, imgs in enumerate(self.train_loader):
                imgs = imgs.to(device=self.device)
                imgs = torch.split(imgs, 1, dim=1)

                for t in range(1, self.max_frame_gap + 1):
                    self.optimizer.zero_grad(set_to_none=True)
                    skip_lr_sch = False

                    img0, imgT, img1 = (
                        imgs[self.max_frame_gap - t].contiguous(),
                        imgs[self.max_frame_gap].contiguous(),
                        imgs[self.max_frame_gap + t].contiguous(),
                    )
                    with autocast("cuda", enabled=self.mixed_precision):
                        rec = self.model(img0, img1)
                        loss = self.loss_fn(rec, imgT)

                    if self.rank == 0:
                        train_loss[f"gap {t}"] += (
                            float(loss.detach().item()) / self.print_freq
                        )

                    old_scale = self.scaler.get_scale()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if old_scale > self.scaler.get_scale():
                        skip_lr_sch = True

                if not skip_lr_sch:
                    self.scheduler.step()

                if self.rank == 0 and self.iter > 0:
                    if self.iter % self.print_freq == 0:
                        self.print_train_loss(train_loss, print_time)
                        train_loss = {key: 0 for key in train_loss.keys()}

                    if self.iter % self.val_freq == 0:
                        val_loss = self.validation()

                    if self.iter % self.save_freq == 0:
                        self.save_model()

                    if self.iter % self.print_freq == 0:
                        print_time = time()           

                self.iter += 1
                
            epoch += 1
            
            if self.do_early_stopping and epoch>=20:
                if self.rank==0:
                    val_loss = self.validation(write=False)
                    if self.early_stopper.early_stop(val_loss):
                        self.log(
                            f"Early stopping training at epoch {epoch}."
                        )
                        break
                sync_nodes(self.is_ddp)

        sync_nodes(self.is_ddp)

        self.log(
            f"Training completed successfully. Total training time = {time()-self.run_stamp:.3f}s"
        )

        cleanup(self.is_ddp)

        sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/default_train.json")
    parser.add_argument("-s", "--seed", type=int, default=-1)
    args = parser.parse_args()

    task = Train(args)
    task.run_training()
