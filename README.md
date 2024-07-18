# CryoSamba: Self-Supervised Deep Volumetric Denoising for Cryo-Electron Tomography Data

This repository contains the segmentation pipeline described in the following publication:

> Jose Inacio Costa-Filho, Liam Theveny, Marilina de Sautu, Tom Kirchhausen<br>[CryoSamba](https://www.biorxiv.org/content/10.1101/2024.07.11.603117v1)<br>
>
> Please cite this publication if you are using this code in your research. For installation, UI, and code setup questions, reach out to [Arkash Jain](https://www.linkedin.com/in/arkashj/) at arkash@tklab.hms.harvard.edu.

## Table of Contents

1. [Overview](#Overview) 📖
   - [CryoSamba_tutorial](#CryoSamba_tutorial) 🛠️
   - [Formatting Data](#Formatting-Data) 📁
   - [Save Folder Structure](#Save-Folder-Structure) 🗂️
   - [Running Trainings](#Running-Trainings) 🚀
   - [Visualization with TensorBoard](#Visualization-with-TensorBoard) 📊
   - [Inference](#Inference) 🔍
2. [UI](#UI) 🖥️
3. [Terminal](#Terminal) 💻
   - [Setup CryoSamba](#Setup-CryoSamba) 🛠️
   - [Training the Model](#Training-the-Model) 🚀
   - [Inference](#Inference) 🔍

## Overview

### CryoSamba_tutorial 🛠️

#### Installing the Conda Environment 🐍

_Note: The script to create the environment has not been updated. Libraries must be installed one by one._

1. Open a terminal and SSH into one of the DGX machines.
2. Run `conda create --name your-env-name python=3.11 -y` to create the environment (replace `your-env-name` with a desired name).
3. Activate the environment with `conda activate your-env-name`.
4. Install PyTorch (for CUDA 11.8):
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
5. Install the remaining libraries:
   ```bash
   pip install tifffile mrcfile easydict loguru tensorboard cupy-cuda11x
   ```
   _Note: Ensure correct spelling (e.g., `loguru`, not `ioguru`)._

### Formatting Data 📁

1. Data must be in one of three formats:

   - A single (3D) .tif file
   - A single .mrc or .rec file
   - A folder containing a sequence of (2D) .tif files, ordered alphanumerically matching the Z-stack order.

   _Note: Ensure consistent zero-fill in file names to maintain proper order (e.g., `frame000.tif` instead of `frame0.tif`)._

### Save Folder Structure 🗂️

1. Recommended folder structure for each experiment:
   ```
   exp-name
   ├── train
   └── inference
   ```
   Running a training session may overwrite the `exp-name/train` folder but won't affect `exp-name/inference`, and vice versa.

### Running Trainings 🚀

1. Go to the `configs` folder inside the project folder and create a new config file. Copy `default_train.json`, rename it, and modify as needed.
2. Mandatory config parameters:
   - `train_dir`: Folder where the checkpoints will be saved (e.g., `exp-name/train`).
   - `data_path`: Filename (for a single 3D file) or folder (for 2D sequence) where the raw data is located.
   - `train_data.max_frame_gap`: Maximum frame gap used for training, as explained in the manuscript.
3. Optional config parameters:
   - `train_data.patch_shape`: X and Y resolution of the patches the model will be trained on (must be a multiple of 32).
   - `train_data.batch_size`: Number of data points loaded into the GPU at once.
   - `train.compile`: If `true`, uses `torch.compile` for faster training.
   - `train.num_iters`: Length of the training run (default is 200k).
4. Other parameters can remain unchanged for now.
5. Run `nvidia-smi` to check available GPUs.
6. (Optional) Use `tmux` to run trainings overnight or run multiple trainings simultaneously.
7. To train on GPUs 0 and 1:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py --config configs/your_config_train.json
   ```
   Adjust `--nproc_per_node` to change the number of GPUs. Use `--seed 1234` for reproducibility.
8. To interrupt training, press `CTRL + C`. You can resume training or start from scratch if prompted.

### Visualization with TensorBoard 📊

1. Open a terminal window inside a graphical interface (e.g., XDesk).
2. Activate the environment and run:
   ```bash
   tensorboard --logdir exp-name/train
   ```
3. In a browser, open `localhost:6006`.
4. Use the slider under `SCALARS` to smooth noisy plots.

### Inference 🔍

1. Create a config file similar to `default_inference.json`.
2. Mandatory config parameters:
   - `train_dir`: Folder where checkpoints were saved (e.g., `exp-name/train`).
   - `data_path`: Filename (for a single 3D file) or folder (for 2D sequence) where the raw data is located.
   - `inference_dir`: Folder where the denoised stack will be saved (e.g., `exp-name/inference`).
   - `inference_data.max_frame_gap`: Maximum frame gap used for inference.
3. Optional config parameters:
   - `inference.patch_shape`: X and Y resolution of the patches (multiple of 32).
   - `inference.TTA`: If `true`, uses test-time augmentation.
   - `inference.compile`: Same as before.
4. Other parameters can remain unchanged for now.
5. To run inference on GPUs 0 and 1:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 inference.py --config configs/your_config_inference.json
   ```
   Adjust `--nproc_per_node` for the number of GPUs. Let us know if you encounter multi-GPU inference errors.
6. To interrupt inference, press `CTRL + C`. You can resume or restart inference if prompted.

## UI 🖥️

From `cryosamba/automate`:

```bash
pip install streamlit
cd automate
streamlit run main.py
```

You can set up the environment, train models, make configs, and run inferences from here.

## Terminal 💻

### CryoSamba Installation 🛠️

Open a terminal window (Powershell if on windows or Terminal if on ubuntu) and navigate to directory where you want to save cryosamba via `cd /path/to/dir`. Then run

```bash
git clone https://github.com/kirchhausenlab/Cryosamba.git
```

in this directory. Once successfully cloned, navigate to the scipts folder via `cd cryosamba/automate/scripts`

##

To setup the environment, run:

```bash
./startup_script_.sh
```

```bash
# In case of permission issues run the command below (OPTIONAL)
chmod u+x ./name_of_file_ending_with.sh
```

This creates a conda environment called `cryosamba` and activates it. In the future, you need to run

```bash
conda activate cryosamba
```

In case of errors, please run `conda init --all && source ~/.bashrc` in your terminal

### Training the Model 🚀

From same directory `automate/scripts`, run:

```bash
./setup_experiment_training.sh
```

When you run this script, it asks you for the name your experiment and if you leave it blank it will generate a default name for you.
The new experiment will be found in the main cryosamba folder. So if you make an experiment called `exp-name`, it will be stored under
`cryosamba/runs/exp-name`. So if you made 5 experiments, it should look like the following:

```bash
cryosamba
├── runs
    ├── exp-name
    ├── exp-2
    ├── exp-test
    ├── exp-
    ├── ...
```

It will generate a config that looks like the following:

```json
{
  "train_data": {
    "max_frame_gap": 6,
    "patch_overlap": [16, 16],
    "split_ratio": 0.95,
    "num_workers": 4,
    "batch_size": 32
  },
  "train": {
    "load_ckpt_path": null,
    "print_freq": 100,
    "save_freq": 1000,
    "val_freq": 1000,
    "warmup_iters": 300,
    "mixed_precision": true,
    "num_iters": 200000
  },
  "optimizer": {
    "lr": 0.0002,
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
  },
  "train_dir": "/nfs/datasync4/inacio/data/denoising/cryosamba/rota/train/",
  "data_path": [
    "/nfs/datasync4/inacio/data/raw_data/cryo/novareconstructions/rotacell_grid1_TS09_ctf_3xBin.rec"
  ]
}
```

The script will prompt you to configure your model, including specifying the locations of your training data and other parameters. You must provide the `train_dir`, `data_path`, and `max_frame_gap`. Then, run:

```bash
./train_data.sh
```

You can select which GPU(s) to use for training.

### Inference 🔍

For inference, navigate to the `inference` folder and run:

```bash
cd ../ && cd inference
./setup_inference.sh
```

The script will prompt you to configure your model for inference. Once configured, run:

```bash
./inference.sh
```

This will execute the inference process based on the specified configuration.
