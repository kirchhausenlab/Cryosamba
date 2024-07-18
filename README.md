# CryoSamba: Self-Supervised Deep Volumetric Denoising for Cryo-Electron Tomography Data

This repository contains the segmentation pipeline described in the following publication:

> Jose Inacio Costa-Filho, Liam Theveny, Marilina de Sautu, Tom Kirchhausen<br>[CryoSamba](https://www.biorxiv.org/content/10.1101/2024.07.11.603117v1)<br>
>
> Please cite this publication if you are using this code in your research. For installation, UI, and code setup questions, reach out to [Arkash Jain](https://www.linkedin.com/in/arkashj/) at arkash@tklab.hms.harvard.edu

## Table of Contents

1. [UI](#ui) 🖥️
2. [Terminal](#terminal) 💻
   - [Installation](#installation) 🛠️
   - [Training](#training) 🚀
   - [Visualization with TensorBoard](#visualization-with-tensorboard) 📈
   - [Inference](#inference) 🔍

## UI 🖥️

From `cryosamba/automate`:

```bash
pip install streamlit
cd automate
streamlit run main.py
```

You can set up the environment, train models, make configs, and run inferences from here.

## Terminal 💻

### Installation 🛠️

Open a terminal window (Powershell if on Windows or Terminal if on Ubuntu) and navigate to the directory where you want to save Cryosamba via `cd /path/to/dir`. Then run

```bash
git clone https://github.com/kirchhausenlab/Cryosamba.git
```

in this directory. Once successfully cloned, navigate to the scipts folder via `cd cryosamba/automate/scripts`

To setup the environment, run:

```bash
./startup_script_.sh
```

```bash
# In case of permission issues run the command below (OPTIONAL)
chmod u+x ./name_of_file_ending_with.sh
```

This creates a conda environment called `cryosamba` and activates it. In the future, you will need to run

```bash
conda activate cryosamba
```
anytime you want to run the CryoSamba again.

In case of errors, try running `conda init --all && source ~/.bashrc` in your terminal.

### Training 🚀

From the same directory `automate/scripts`, run:

```bash
./setup_experiment_training.sh
```

The script asks you to enter the following parameters:

- Experiment name: it will create the following folder structure

```bash
cryosamba
├── runs
    ├── exp-name
       ├── train
       ├── inference
       train_config.json
```

- Data path: it must be either

  - The full path to a single (3D) .tif, .mrc or .rec file, or
  - The full path to a folder containing a sequence of (2D) .tif files, ordered alphanumerically matching the Z-stack order.

  _Note: Ensure consistent zero-fill in file names to maintain proper order (e.g., `frame000.tif` instead of `frame0.tif`)._

- Max frame gap: explained in the manuscript. We empirically set values of 3, 6 and 10 for data at voxel resolutions of 15.72Å, 7.86Å and 2.62Å, respectively. For different resolutions, try a reasonable interpolated value between the reference ones.
- Number of iterations
- Batch Size

The generated `train_config.json` file will contain all parameters for training the model and will look like the following:

```json
{
  "train_dir": "/path/to/dir/cryosamba/runs/exp-name/train",
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

If you want to change other parameters, edit the `.json` file directly. In `installation_instructions.md` we provide a full explanation of all config parameters.

To start training the model, run the command below from the same folder `automate/scripts`

```bash
./train_data.sh
```

To interrupt training, press CTRL + C. You can resume training or start from scratch if prompted.

### Visualization with TensorBoard 📊

TensorBoard can be used to monitor the progress of the training losses.

1. Open a terminal window inside a graphical interface (e.g., XDesk).
2. Activate the environment and run:
   ```bash
   tensorboard --logdir /path/to/dir/cryosamba/runs/exp-name/train
   ```
3. In a browser, open `localhost:6006`.
4. Use the slider under `SCALARS` to smooth noisy plots.

### Inference 🔍

From the same directory `automate/scripts`, run:

```bash
./setup_inference.sh
```

The script asks you to enter the following parameters:

- Experiment name: same as in training (should be an existing one)
- Data path: same as in training
- Max frame gap: usually twice the value used for training
- TTA: whether to use Test-Time Augmentation or not (see manuscript)

The generated `inference_config.json` file will contain all parameters for running inference and will look like the following:

```json
{
  "train_dir": "/path/to/dir/cryosamba/runs/exp-name/train",
  "data_path": "/path/to/file/volume.mrc",
  "inference_dir": "/path/to/dir/cryosamba/runs/exp-name/inference",
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
    "compile": false
  }
}
```

If you want to change other parameters, edit the `.json` file directly.

To start inference, run the command below from the same folder `automate/scripts`

```bash
./inference.sh
```

To interrupt the process, press CTRL + C. You can resume or start from scratch if prompted.

The final denoised volume will be located at `/path/to/dir/cryosamba/runs/exp-name/inference`. It will be either a file named `result.tif`, `result.mrc`, `result.rec` or a folder named `result`.

