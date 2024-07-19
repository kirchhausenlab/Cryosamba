# Manual installation instructions

## Table of Contents

1. [Installation](#installation) 🐍
2. [Training](#training) 
   - [Setup Training](#setup-training) 🛠️
   - [Run Training](#run-training) 🚀
   - [Visualization with TensorBoard](#visualization-with-tensorboard) 📈
3. [Inference](#inference) 
   - [Setup Inference](#setup-inference) 🛠️
   - [Run Inference](#run-inference) 🚀

## Installation

1. Open a terminal window and run `conda create --name your-env-name python=3.11 -y` to create the environment (replace `your-env-name` with a desired name).
2. Activate the environment with `conda activate your-env-name`. In the future, you will have to activate the environment anytime you want to use CryoSamba.
3. Install PyTorch (for CUDA 11.8):
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
4. Install the remaining libraries:
   ```bash
   pip install tifffile mrcfile easydict loguru tensorboard cupy-cuda11x
   ```
5. Navigate to the directory where you want to save the Cryosamba code (via `cd /path/to/dir`). Then run
```bash
git clone https://github.com/kirchhausenlab/Cryosamba.git
```
in this directory. If you only have access to a Cryosamba `.zip` file instead, simply extract it there.

## Training

### Setup Training

Create a `your_train_config.json` config file somewhere. Use as default the template below:

```json
{
  "train_dir": "/path/to/dir/Cryosamba/runs/exp-name/train",
  "data_path": ["/path/to/file/volume.mrc"],
  "train_data": {
    "max_frame_gap": 6,
    "patch_shape": [256, 256],
    "patch_overlap": [16, 16],
    "split_ratio": 0.95,
    "batch_size": 32,
    "num_workers": 4
  },
  "train": {
    "load_ckpt_path": null,
    "print_freq": 100,
    "save_freq": 1000,
    "val_freq": 1000,
    "num_iters": 200000,
    "warmup_iters": 300,
    "mixed_precision": true,
    "compile": false
  },
  "optimizer": {
    "lr": 2e-4,
    "lr_decay": 0.99995,
    "weight_decay": 0.0001,
    "epsilon": 1e-8,
    "betas": [0.9, 0.999]
  },
  "biflownet": {
    "pyr_dim": 24,
    "pyr_level": 3,
    "corr_radius": 4,
    "kernel_size": 3,
    "warp_type": "soft_splat",
    "padding_mode": "reflect",
    "fix_params": false
  },
  "fusionnet": {
    "num_channels": 16,
    "padding_mode": "reflect",
    "fix_params": false
  }
}
```

Explanation of parameters:

- `train_dir`: Folder where the checkpoints will be saved (e.g., `exp-name/train`).
- `data_path`: full path to a single (3D) .tif, .mrc or .rec file, or full path to a folder containing a sequence of (2D) .tif files, ordered alphanumerically matching the Z-stack order. You can train on multiple volumes by including the paths as elements of a list.
- `train_data`: parameters related to the raw data used for training
   - `max_frame_gap`: Maximum frame gap used for training (see manuscript). For our data, we used values of 3, 6 and 10 for resolutions of 15.72, 7.86 and 2.62 angstroms/voxel, respectively.
   - `patch_shape`: X and Y resolution of the patches the model will be trained on (must be a multiple of 32).
   - `patch_overlap`: overlap (on X and Y) between consecutive patches (see manuscript).
   - `split_ratio`: train and validation data split ratio. Must be a float between 0 and 1. E.g., 0.95 means that 95% of the data will be assigned for training and 5% for validation.
   - `batch_size`: Number of data points loaded into the GPU at once.
   - `num_workers`: Number of simultaneous CPU workers used by the Pytorch Dataloader.
- `train`: parameters related to the training routine
   - `load_ckpt_path`: `null` to start a training run from scratch, or the path to a (`.pt` or `.pth`) model checkpoint if you want to start from a pretrained model.
   - `print_freq`: frequency of the print statements in number of iterations.
   - `save_freq`: frequency of model checkpoint saving in number of iterations.
   - `val_freq`: frequency validation runs in number of iterations.
   - `num_iters`: Length of the training run (default is 200k iterations).
   - `warmup_iters`: number of iterations for the learning rate warmu-up.
   - `mixed_precision`: if `true`, uses mixed precision training.
   - `compile`: If `true`, uses `torch.compile` for faster training (might lead to errors, and has a few-minutes overhead time before the training iterations).
- `optimizer`: parameters related to the optimization algorithm
   - `lr`: base learning rate.
   - `lr_decay`: multiplicative factor for the learning rate decay.
   - `weight_decay`: weight decay for the AdamW optimizer.
   - `epsilon`: epsilon for the AdamW optimizer.
   - `betas`: betas for the AdamW optimizer.
- `biflownet`: parameters related to the Bi-Directional Optical Flow module (see manuscript and EBME paper)
   - `pyr_dim`: base number of channels of the Feature Pyramid and Flow Estimator networks' layers.
   - `pyr_level`: number of pyramid levels of the Feature Pyramid network.
   - `corr_radius`: radius of the correlation volume function.
   - `kernel_size`: kernel size of the biflownet convolutional layers.
   - `warp_type`: type of Optical Flow warping (`soft_splat`, `avg_splat`, `fw_splat` or `backwarp`)
   - `padding_mode`: padding mode of biflownet convolutional layers.
   - `fix_params`: set to true in order to fix biflownet's weights and disable learning.
- `fusionnet`:
   - `num_channels`: base number of channels of fusionnet layers.
   - `padding_mode`: padding mode of fusionnet convolutional layers.
   - `fix_params`: set to true in order to fix fusionnet's weights and disable learning.
 
Recommended folder structure for each experiment:
```
exp-name
├── train
└── inference
```
Running a training session may overwrite the `exp-name/train` folder but won't affect `exp-name/inference`, and vice versa.
 
### Run Training

1. In the terminal, run `nvidia-smi` to check available GPUs. For example, if you have 8 GPUs they will be numbered from 0 to 7.
8. To train on GPUs 0 and 1, go to the CryoSamba folder and run:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py --config path/to/your_train_config.json
   ```
   Adjust `--nproc_per_node` to change the number of GPUs. Use `--seed 1234` for reproducibility.
9. To interrupt training, press `CTRL + C`. You can resume training or start from scratch if prompted.

### Visualization with TensorBoard

1. Open a terminal window inside a graphical interface (e.g., a regular desktop computer, Chrome Remote Desktop, XDesk).
2. Activate the environment and run:
   ```bash
   tensorboard --logdir path/to/exp-name/train
   ```
3. In a browser, open `localhost:6006`.
4. Use the slider under `SCALARS` to smooth noisy plots.

## Inference

### Setup Inference

Create a `your_inference_config.json` config file somewhere. Use as default the template below:

```json
{
    "train_dir": "/path/to/dir/Cryosamba/runs/exp-name/train",
    "data_path": "/path/to/file/volume.mrc",
    "inference_dir": "/path/to/dir/Cryosamba/runs/exp-name/inference",
    "inference_data": {
        "max_frame_gap": 12,
        "patch_shape": [256, 256],
        "patch_overlap": [16, 16],
        "batch_size": 32,
        "num_workers": 4
    },
    "inference": {
        "output_format": "same",
        "load_ckpt_name": null,
        "pyr_level": 3,
        "TTA": true,
        "mixed_precision": true,
        "compile": true
    }
}
```

Explanation of parameters:

- `train_dir`: Folder from which the checkpoints will be loaded.
- `data_path`: full path to a single (3D) .tif, .mrc or .rec file, or full path to a folder containing a sequence of (2D) .tif files, ordered alphanumerically matching the Z-stack order.
- `inference_dir`: Folder where the denoised data will be saved (e.g., `exp-name/inference`).
- `inference_data`: parameters related to the raw data used for inference
   - `max_frame_gap`: maximum frame gap used for inference (see manuscript). For our data, we used values of 6, 12 and 20 for resolutions of 15.72, 7.86 and 2.62 angstroms/voxel, respectively.
   - `patch_shape`: X and Y resolution of the patches the model will be trained on (must be a multiple of 32).
   - `patch_overlap`: overlap (on X and Y) between consecutive patches (see manuscript).
   - `batch_size`: Number of data points loaded into the GPU at once.
   - `num_workers`: Number of simultaneous CPU workers used by the Pytorch Dataloader.
- `inference`: parameters related to the inference routine
   - `output_format`: `"same"` to save the denoised result in the same format as the input raw data. Otherwise, specify either `"tif_file"`, `"mrc_file"`, `"rec_file"` or `"tif_sequence"`.
   - `load_ckpt_path`: `null` to load model weights from `train-dir/last.pt`, otherwise use the path for a custom model checkpoint.
   - `pyr_level`: number of pyramid levels of the Feature Pyramid network of the biflownet.
   - `TTA`: if `true`, uses Test-Time Augmentation (see manuscript), for slightly better results at the cost of longer inference times.
   - `mixed_precision`: if `true`, uses mixed precision training.
   - `compile`: If `true`, uses `torch.compile` for faster inference (might lead to errors, and has a few-minutes overhead time before the inference iterations).
 

### Run Inference

1. In the terminal, run `nvidia-smi` to check available GPUs. For example, if you have 8 GPUs they will be numbered from 0 to 7.
2. To run inference on GPUs 0 and 1, go to the CryoSamba folder and run:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 inference.py --config path/to/your_inference_config.json
   ```
   Adjust `--nproc_per_node` to change the number of GPUs. Use `--seed 1234` for reproducibility.
3. To interrupt inference, press `CTRL + C`. You can resume or start from scratch if prompted.
4. The final denoised volume will be located at `/path/to/dir/runs/exp-name/inference`. It will be either a file named `result.tif`, `result.mrc`, `result.rec` or a folder named `result`.
 
