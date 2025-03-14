"""
The test_sample folder has some sample data produced by running 500 iterations of cryosamba on a DGX A100 GPU with a batch size of
16 and max frame gap of 3. The following test recreates that scenario for users and will take around 15 minutes to run (both train and inference)
Once you have the training and inference done, navigate to the test_rotacell folder, where you will find our sample results. Compare
what you get by running this test to our sample as a sanity check for cryosamba's validity

Download the ndc10gfp_g7_l1_ts_002_ctf_6xBin.rec file from dropbox and put it in the cryosamba folder before running the tests
"""

import json
import os
import subprocess
import typer
from pathlib import Path
import time


def main():

    start_time = time.time()
    curr_path = Path(__name__).resolve().parent
    test_file = f"{curr_path}/ndc10gfp_g7_l1_ts_002_ctf_6xBin.rec"

    train_config = {
        "train_dir": "test_sample/train",
        "data_path": test_file,
        "train_data": {
            "max_frame_gap": 3,
            "patch_overlap": [16, 16],
            "patch_shape": [256, 256],
            "split_ratio": 0.95,
            "batch_size": 16,
            "num_workers": 4,
        },
        "train": {
            "num_iters": 500,
            "load_ckpt_path": None,
            "print_freq": 100,
            "save_freq": 1000,
            "val_freq": 1000,
            "warmup_iters": 300,
            "mixed_precision": True,
            "compile": False,
        },
        "optimizer": {
            "lr": 2e-4,
            "lr_decay": 0.99995,
            "weight_decay": 0.0001,
            "epsilon": 1e-08,
            "betas": [0.9, 0.999],
        },
        "biflownet": {
            "pyr_dim": 24,
            "pyr_level": 3,
            "corr_radius": 4,
            "kernel_size": 3,
            "warp_type": "soft_splat",
            "padding_mode": "reflect",
            "fix_params": False,
        },
        "fusionnet": {
            "num_channels": 16,
            "padding_mode": "reflect",
            "fix_params": False,
        },
    }

    # Generate inference config
    inference_config = {
        "train_dir": "test_sample/train",
        "data_path": test_file,
        "inference_dir": "test_sample/inference",
        "inference_data": {
            "max_frame_gap": 6,
            "patch_shape": [256, 256],
            "patch_overlap": [16, 16],
            "batch_size": 16,
            "num_workers": 4,
        },
        "inference": {
            "output_format": "same",
            "load_ckpt_name": None,
            "pyr_level": 3,
            "mixed_precision": True,
            "tta": True,
            "compile": False,
        },
    }

    # Save train config to JSON
    train_config_path = curr_path / "test_sample" / "train_config.json"
    with open(train_config_path, "w") as f:
        json.dump(train_config, f, indent=4)

    # Save inference config to JSON
    inference_config_path = curr_path / "test_sample" / "inference_config.json"
    with open(inference_config_path, "w") as f:
        json.dump(inference_config, f, indent=4)

    flag = True
    while time.time() - start_time < 15000:
        cmd = f"CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py --config {train_config_path}"
        subprocess.run(cmd, shell=True, text=True)
        flag = False
        print("finished execution!")
        break
    if flag:
        print("too slow, check your compute!!")

    cmd = f"CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 inference.py --config {inference_config_path}"
    subprocess.run(cmd, shell=True, text=True)


if __name__ == "__main__":
    typer.run(main)
