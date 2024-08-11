# CryoSamba: Self-Supervised Deep Volumetric Denoising for Cryo-Electron Tomography Data

<img src="https://github.com/kirchhausenlab/Cryosamba/blob/main/denoising_comparison.gif" width="800"/>

This repository contains the denoising pipeline described in the following publication:

> Jose Inacio Costa-Filho, Liam Theveny, Marilina de Sautu, Tom Kirchhausen<br>[CryoSamba: Self-Supervised Deep Volumetric Denoising for Cryo-Electron Tomography Data](https://www.biorxiv.org/content/10.1101/2024.07.11.603117v1)<br>
>
> Please cite this publication if you are using this code in your research. For installation, UI, and code setup questions, reach out to [Arkash Jain](https://www.linkedin.com/in/arkashj/) at arkash@tklab.hms.harvard.edu

❗**WARNING**❗ CryoSamba is written for machines with either a **Linux** or **Windows** operating system and a **CUDA capable GPU**. **MacOS is not supported**.

❗**WARNING**❗Make sure you have **CUDA** drivers installed and updated on your machine. CryoSamba requires **CUDA 11** to run. Support for CUDA 12 will be added soon. Refer to [Instructions for Setting Up CUDA](#instructions-for-setting-up-cuda) for more support.

❗**WARNING**❗CryoSamba is based on the **Python** programming language. Prior experience with Python is desirable but not strictly required.

These instructions are meant to be read **carefully** and **line by line**. Arbitrarily skipping lines or blindly copy-pasting code snippets will likely lead to errors.

💥**IMPORTANT**💥 CryoSamba accepts as input (.mrc, .rec, .tif) single 3D files or sequences of 2D (.tif) files. For single files, the "data path" must directly reference the files, while for tif sequences the "data path" should reference the folder containing the sequence. For example, use `path/to/sample_data.rec` or `path/to/tif_folder". Not referencing the input data properly will lead to errors.


## Table of Contents

1. [Overview](#overview) 🌐
2. [CLI Tool](#cli-tool) 📟
   - [Installation](#installation) 🛠️
   - [Using the tool](#using-the-tool) 🔨
3. [Terminal](#terminal) 💻
   - [Installation](#installation) 🛠️
   - [Training](#training) 🚀
   - [Visualization with TensorBoard](#visualization-with-tensorboard) 📈
   - [Inference](#inference) 🔍
4. [UI](#ui) 🎮
5. [Instructions for Setting Up CUDA](#instructions-for-setting-up-cuda)

## Overview

CryoSamba can be run via Terminal, via a CLI Tool or via a UI (work in progress).

If you have access to a graphical interface, try the [UI](#ui).

If you do not have access to a graphical interface, try the [CLI Tool](#cli-tool). This is usually the case if you want to run CryoSamba on your university's HPC. **This the recommended option!**.

If you are a more experience programmer and comfortable with using the Terminal, try our "raw" [Terminal](#terminal) instructions.

Finally, if you want to use CryoSamba on Windows, have a deeper understanding of the source code, change the optional parameters, or alter/use the code for your research, refer to the [advanced instructions](https://github.com/kirchhausenlab/Cryosamba/blob/main/advanced_instructions.md).

Our goal is to make CryoSamba as accessible as possible.

## CLI Tool

❗**WARNING**❗ These instructions require you to know how to open a terminal window on your computer, how to navigate through folders and how to copy files around.

Note: these instructions are designed for machines with a **Linux** operating system. For **Windows**, refer to the [advanced instructions](https://github.com/kirchhausenlab/Cryosamba/blob/main/advanced_instructions.md).

### Installation

1. Open a Terminal window and navigate to the directory where you want to save the Cryosamba code via `cd /path/to/dir`.

Note: the expression `/path/to/dir` is not meant to be copy-pasted as it is. It is a general expression which means that you should replace it with the actual path to the desired directory in your own computer. Since we do not have access to your computer, we cannot give you the exact expression to copy-paste. This expression will appear several times throughout these instructions.

2) If you received CryoSamba via a zip file, run
```bash
unzip path/to/Cryosamba.zip
```
in this directory. **Otherwise**, run
```bash
git clone https://github.com/kirchhausenlab/Cryosamba.git
```
These two options are **mutually exclusive**. 

3. Once successfully cloned/unzipped, navigate to the scripts folder via `cd path/to/Cryosamba/automate/scripts`

4. To setup the environment, run:
```bash
chmod -R u+x *.sh
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
anytime you want to run CryoSamba again.

In case of errors, try running `conda init --all && source ~/.bashrc` in your terminal.

### Using the Tool

From the directory `CryoSamba/automate`, run
```bash
python run_cryosamba_cli.py
```
and follow the instructions that appear on the Terminal window.

## Terminal

❗**WARNING**❗ These instructions require you to know how to open a terminal window on your computer, how to navigate through folders and how to copy files around.

Note: these instructions are designed for machines with a **Linux** operating system. For **Windows**, refer to the [advanced instructions](https://github.com/kirchhausenlab/Cryosamba/blob/main/advanced_instructions.md).

### Installation

1. Open a Terminal window and navigate to the directory where you want to save the Cryosamba code via `cd /path/to/dir`.

Note: the expression `/path/to/dir` is not meant to be copy-pasted as it is. It is a general expression which means that you should replace it with the actual path to the desired directory in your own computer. Since we do not have access to your computer, we cannot give you the exact expression to copy-paste. This expression will appear several times throughout these instructions.

2) If you received CryoSamba via a zip file, run
```bash
unzip path/to/Cryosamba.zip
```
in this directory. **Otherwise**, run
```bash
git clone https://github.com/kirchhausenlab/Cryosamba.git
```
These two options are **mutually exclusive**. 

3. Once successfully cloned/unzipped, navigate to the scripts folder via `cd path/to/Cryosamba/automate/scripts`

4. To setup the environment, run:
```bash
chmod -R u+x *.sh
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
anytime you want to run CryoSamba again.

In case of errors, try running `conda init --all && source ~/.bashrc` in your terminal.

### Training

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

- Max frame gap: explained in the manuscript. We empirically set values of 3, 6 and 10 for data at voxel resolutions of 15.72Å, 7.86Å and 2.62Å, respectively. For different resolutions, try a reasonable value interpolated from the reference ones.
- Number of iterations: for how many iterations the training session will run
- Batch Size: number of data points passed at once to the GPUs. Higher number leads to faster training, but the whole batch might not fit into your GPU's memory, leading to **out-of-memory errors**. If you're getting these, try to decrease the batch size until they disappear. This number should be a multiple of two.

The generated `train_config.json` file will contain all parameters for training the model. If you want to change other parameters, edit the `.json` file directly. In [advanced instructions](https://github.com/kirchhausenlab/Cryosamba/blob/main/advanced_instructions.md) we provide a full explanation of all config parameters.

To start training the model, run the command below from the same folder `automate/scripts`

```bash
./train_data.sh
```

To interrupt training, press CTRL + C. You can resume training or start from scratch if prompted.

Training will run until the maximum number of iterations is reached. However, training and validation losses might converge/stabilize before that, at which point you can safely halt the process and save time and money on your electricity bill. In order to monitor the losses' progress you can: 1) see the logs printed on your screen, 2) see the `runtime.log` file inside your training folder, or 3) visualize their plots with TensorBoard. 

**The output of the training run will be checkpoint files containing the trained model weights**. There is no denoised data output at this point yet. You can used the trained model weights to run inference on your data and then get the denoised outputs.

### Visualization with TensorBoard (OPTIONAL)

TensorBoard can be used to monitor the progress of the training losses.

1. Open a terminal window inside a graphical interface (e.g., XDesk).
2. Activate the environment and run:
   ```bash
   tensorboard --logdir /path/to/dir/Cryosamba/runs/exp-name/train
   ```
3. In a browser, open `localhost:6006`.
4. Use the slider under `SCALARS` to smooth noisy plots.

### Inference

From the same directory `automate/scripts`, run:

```bash
./setup_inference.sh
```

The script asks you to enter the following parameters:

- Experiment name: same as in training (should be an existing one)
- Data path: same as in training
- Max frame gap: usually twice the value used for training
- TTA: whether to use Test-Time Augmentation or not (see manuscript)

The generated `inference_config.json` file will contain all parameters for running inference. If you want to change other parameters, edit the `.json` file directly.

To start inference, run the command below from the same folder `automate/scripts`

```bash
./run_inference.sh
```

To interrupt the process, press CTRL + C. You can resume or start from scratch if prompted.

**The output of the inference run will be the final denoised volume**, located at `/path/to/dir/runs/exp-name/inference`. It will be either a file named `result.tif`, `result.mrc`, `result.rec` or a folder named `result`.

You can simply open the final denoised volume in your preferred data visualization/processing software and check how it looks like.

## UI

### PLEASE WATCH THE VIDEOS IN THE GITHUB (move_to_remote_server.mp4, install_and_startup.mp4 and How_to_run.mp4 to see an end-to-end example of running CryoSamba)

From `Cryosamba/automate`:

```bash
pip install streamlit
cd automate
chmod -R u+x *.sh
streamlit run main.py
```

You can set up the environment, train models, make configs, and run inferences from here.

## Instructions for Setting Up CUDA

If it appears that your machine is unable to locate the CUDA driver, which is typically found under `/usr/bin/`, please follow the steps below after identifying the path for CUDA on your machine:

1. **Set the CUDA Home Environment Variable**

   Run the following command, replacing the path with the correct one for your CUDA installation:

   ```bash
   export CUDA_HOME=/path/to/your/cuda

   ```

   For example:

   ```bash
   export CUDA_HOME=/uufs/pathto_/sys/modulefiles/CHPC-r8/Core/cuda/12.2.0.lua

   ```

2. **Ensure CUDA 11 is Installed**

   Verify that CUDA version 11 is installed on your system. If it is not, please install it according to the official NVIDIA documentation.

By following these steps, your machine should be able to locate and use the CUDA driver, allowing you to proceed with your work.
