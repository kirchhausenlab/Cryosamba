import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import argparse
from time import time

import torch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from core.model import get_model
from core.dataset import get_dataloader
from core.utils.utils import setup_run, remove_file, load_json, logger_info, listify
from core.utils.data_utils import (
    get_data,
    unpad3D,
    save_data,
    denormalize_imgs,
    get_overlap_pad,
)
from core.utils.torch_utils import (
    setup_DDP,
    sync_nodes,
    count_model_params,
    cleanup,
    load_ckpt,
)


class Inference:
    def __init__(self, args):
        self.run_stamp = time()

        self.world_size, self.rank, self.device = setup_DDP(args.seed)
        self.is_ddp = self.world_size > 1
        cfg = load_json(args.config)

        if self.rank == 0:
            setup_run(cfg, mode="inference")
        sync_nodes(self.is_ddp)

        self.log(
            f"Using seed number {args.seed}"
            if args.seed != -1
            else f"No random seed was set"
        )
        self.log(f"# of processes = {self.world_size}")

        if os.path.exists(cfg.train_dir):
            cfg_train = load_json(os.path.join(cfg.train_dir, "config.json"))
        else:
            raise ValueError(f"Checkpoint dir {cfg.train_dir} does not exist")

        cfg_train.biflownet.pyr_level = cfg.inference.pyr_level
        cfg_train.train_data = cfg.inference_data

        # init params

        self.overlap_pad = get_overlap_pad(
            cfg.inference_data.patch_overlap, self.device
        )
        self.TTA = cfg.inference.TTA
        self.mixed_precision = cfg.inference.mixed_precision
        self.inference_dir = cfg.inference_dir
        self.output_format = cfg.inference.output_format
        self.max_frame_gap = cfg.inference_data.max_frame_gap
        self.batch_size = cfg.inference_data.batch_size
        self.output_temp_name = os.path.join(cfg.inference_dir, "temp.dat")
        self.compile = cfg.inference.compile

        # Init model

        ckpt_name = (
            "last"
            if cfg.inference.load_ckpt_name is None
            else cfg.inference.load_ckpt_name
        )
        ckpt_path = os.path.join(cfg.train_dir, ckpt_name + ".pt")
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Model checkpoint {ckpt_path} does not exist")

        self.model = get_model(
            cfg_train, self.device, is_ddp=self.is_ddp, compile=self.compile
        )
        self.model = load_ckpt(
            ckpt_path, self.model, is_ddp=self.is_ddp, compile=self.compile
        )[0]
        self.model.eval()

        self.log(f"# of model parameters = {count_model_params(self.model)[1]}")
        self.log(f"Using trained weights from {ckpt_path}")

        # Init dataloaders

        cfg.data_path = listify(cfg.data_path)
        data_list, metadata_list = list(
            zip(*[get_data(path) for path in cfg.data_path])
        )
        self.inference_loader = get_dataloader(
            cfg.inference_data,
            data_list,
            metadata_list,
            split="test",
            is_ddp=self.is_ddp,
        )
        self.metadata = metadata_list[0]

        self.log(f"# of data samples = {len(self.inference_loader)}")

        # Make output temporary file

        self.make_output_temp_file()
        sync_nodes(self.is_ddp)
        self.output_array = np.memmap(
            self.output_temp_name,
            dtype=self.metadata["dtype"],
            mode="r+",
            shape=self.metadata["shape"],
        )

    def log(self, message):
        return logger_info(self.rank, message)

    def make_output_temp_file(self):
        if not os.path.exists(self.output_temp_name):
            if self.rank == 0:
                output_array = np.memmap(
                    self.output_temp_name,
                    dtype=self.metadata["dtype"],
                    mode="w+",
                    shape=self.metadata["shape"],
                )
                z_border = self.max_frame_gap + 1
                output_array[0:z_border] = self.metadata["mean"]
                output_array[-z_border:] = self.metadata["mean"]
                output_array.flush()

    def process_crop_params(self, crop_params):
        coords, border_pad = torch.split(crop_params, 3, dim=1)
        residual_pad = self.overlap_pad * (self.overlap_pad > border_pad)
        residual_pad[..., 1] *= -1

        pad = torch.maximum(border_pad, self.overlap_pad)
        out_coords = coords + residual_pad
        z = coords[:, 0, 0] + self.max_frame_gap
        return pad, out_coords, z

    def skip_iter(self, imgs, z, out_coords):
        output_mimmax = min(
            [
                self.output_array[
                    z[j],
                    out_coords[j, 1, 0] : out_coords[j, 1, 1],
                    out_coords[j, 2, 0] : out_coords[j, 2, 1],
                ].max()
                for j in range(imgs.shape[0])
            ]
        )
        return True if output_mimmax != 0.0 else False

    def TTA_transforms(self, x):
        if self.TTA:
            return [
                x[0],
                x[1].flip(dims=[-1]),
                x[2].flip(dims=[-2]),
                x[3].flip(dims=[-1, -2]),
            ]
        else:
            return x

    def samba(self, img0, imgT, img1):
        rec_minus = self.model(img0, imgT)
        rec_plus = self.model(imgT, img1)
        rec = self.model(rec_minus, rec_plus)
        return rec

    def inference_fn(self, img0, imgT, img1):
        img0 = [img0, img0, img0, img0] if self.TTA == True else [img0]
        imgT = [imgT, imgT, imgT, imgT] if self.TTA == True else [imgT]
        img1 = [img1, img1, img1, img1] if self.TTA == True else [img1]

        img0 = self.TTA_transforms(img0)
        imgT = self.TTA_transforms(imgT)
        img1 = self.TTA_transforms(img1)

        recs = [self.samba(img0[i], imgT[i], img1[i]) for i in range(len(img0))]
        recs = self.TTA_transforms(recs)

        recs = torch.cat(recs, dim=1).mean(dim=1, keepdim=True)

        return recs

    def run_inference(self):

        for i, [imgs, crop_params] in enumerate(self.inference_loader):
            iter_time = time()

            imgs, crop_params = imgs.to(device=self.device), crop_params.to(
                device=self.device
            )
            pad, out_coords, z = self.process_crop_params(crop_params)
            if self.skip_iter(imgs, z, out_coords):
                continue

            imgs = torch.split(imgs, 1, dim=1)

            recs = []
            for t in range(1, self.max_frame_gap + 1):
                img0, imgT, img1 = (
                    imgs[self.max_frame_gap - t].contiguous(),
                    imgs[self.max_frame_gap].contiguous(),
                    imgs[self.max_frame_gap + t].contiguous(),
                )

                with autocast(enabled=self.mixed_precision):
                    with torch.inference_mode():
                        rec = self.inference_fn(img0, imgT, img1)
                recs.append(rec)
            rec = torch.cat(recs, dim=1).mean(dim=1, keepdim=True)
            rec = denormalize_imgs(rec, params=self.metadata)
            rec = rec.cpu().detach().numpy().astype(self.metadata["dtype"])

            for j in range(rec.shape[0]):
                self.output_array[
                    z[j],
                    out_coords[j, 1, 0] : out_coords[j, 1, 1],
                    out_coords[j, 2, 0] : out_coords[j, 2, 1],
                ] = unpad3D(rec[j], pad[j])

            self.log(
                f"Iter {i}/{len(self.inference_loader)}, Elapsed time = {(time()-iter_time):.3f}"
            )
            self.output_array.flush()

        sync_nodes(self.is_ddp)
        if self.rank == 0:
            self.log(f"Saving results")
            save_data(
                path=self.inference_dir,
                name="result",
                data=self.output_array,
                metadata=self.metadata,
                output_format=self.output_format,
            )
        sync_nodes(self.is_ddp)

        if self.rank == 0:
            remove_file(self.output_temp_name)

        self.log(
            f"Inference completed successfully. Total inference time = {time()-self.run_stamp:.3f}s"
        )

        cleanup(self.is_ddp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/default_inference.json")
    parser.add_argument("-s", "--seed", type=int, default=-1)
    args = parser.parse_args()

    task = Inference(args)
    task.run_inference()
