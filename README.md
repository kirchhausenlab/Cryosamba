# CryoSamba: Self-Supervised Deep Volumetric Denoising for Cryo-Electron Tomography Data

<img src="https://github.com/kirchhausenlab/Cryosamba/blob/main/assets/denoising_comparison.gif" width="800"/>

This repository contains the denoising pipeline described in the following publication:

> Jose Inacio Costa-Filho, Liam Theveny, Marilina de Sautu, Tom Kirchhausen<br>[CryoSamba: Self-Supervised Deep Volumetric Denoising for Cryo-Electron Tomography Data](https://www.biorxiv.org/content/10.1101/2024.07.11.603117v1)<br>
>
> Please cite this publication if you are using this code in your research. For installation, UI, and code setup questions, reach out to [Arkash Jain](https://www.linkedin.com/in/arkashj/) at arkash@tklab.hms.harvard.edu

❗**WARNING**❗ CryoSamba is written for machines with either a **Linux** or **Windows** operating system and a **CUDA capable GPU**. **MacOS is not supported**.

❗**WARNING**❗Make sure you have **CUDA** drivers installed and updated on your machine. CryoSamba requires **CUDA 11 or higher** to run. Refer to [Troubleshooting](#troubleshooting) for more support.

## Table of Contents

1. [Overview](#overview) 🌐
2. [Terminal](#terminal) 💻
   - [Installation](#installation) 🛠️
   - [Usage](#usage) 🔨
3. [(OPTIONAL) Visualization with TensorBoard](#visualization-with-tensorboard) 📈
4. [(Work in Progress) UI](#ui) 🎮
5. [Troubleshooting](#troubleshooting) 🎯

## Overview

CryoSamba can be run via [Terminal](#terminal). We also provide a [User Interface (UI)](#ui) implementation, which is currently **experimental**.

If you want to use CryoSamba on Windows, have a deeper understanding of the source code, change the optional parameters, or alter/use the code for your research, refer to the [advanced instructions](https://github.com/kirchhausenlab/Cryosamba/blob/main/advanced_instructions.md).

## Terminal

❗**WARNING**❗ These instructions require you to know how to open a terminal window on your computer, how to navigate through folders and how to copy files around.

Note: these instructions are designed for machines with a **Linux** operating system. For **Windows**, refer to the [advanced instructions](https://github.com/kirchhausenlab/Cryosamba/blob/main/advanced_instructions.md).

### Installation

1. Open a Terminal window and navigate to the directory where you want to save the Cryosamba code via
```bash
cd /path/to/dir
```

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

3. Once successfully cloned/unzipped, navigate to the CryoSamba folder via `cd Cryosamba` and run
```bash
automate/scripts/startup_script.sh
```

```bash
# In case of permission issues run the command below (OPTIONAL)
chmod -R u+x automate/scripts/*.sh
```

This creates a conda environment called `cryosamba` and activates it. 

4. In the future, you will need to navigate to the CryoSamba folder (via `cd /path/to/dir/Cryosamba`) and activate the environment
```bash
conda activate cryosamba
```
anytime you want to run CryoSamba again.

In case of errors, try running `conda init --all && source ~/.bashrc` in your terminal.

### Usage

From the CryoSamba directory, run
```bash
python automate/run_cryosamba_cli.py
```
and follow the instructions that appear on the Terminal window.

## Visualization with TensorBoard

**This step is OPTIONAL**

TensorBoard can be used to monitor the progress of the training losses.

1. Open a terminal window inside a graphical interface (e.g., your desktop computer, or XDesk).
2. Activate the environment and run:
   ```bash
   tensorboard --logdir /path/to/dir/Cryosamba/runs/exp-name/train
   ```
3. In a browser, open `localhost:6006`.
4. Use the slider under `SCALARS` to smooth noisy plots.

## UI

🚧**Work in Progress**🚧

### PLEASE WATCH THE VIDEOS IN THE GITHUB (move_to_remote_server.mp4, install_and_startup.mp4 and How_to_run.mp4 to see an end-to-end example of running CryoSamba)

From `Cryosamba/automate`:

```bash
pip install streamlit
cd automate
chmod -R u+x *.sh
streamlit run main.py
```

You can set up the environment, train models, make configs, and run inferences from here.

## Troubleshooting

### Instructions for Setting Up CUDA

If it appears that your machine is unable to locate the CUDA driver, which is typically found under `/usr/bin/`, please follow the steps below after identifying the path for CUDA on your machine:

1. **Set the CUDA Home Environment Variable**

   Run the following command, replacing the path with the correct one for your CUDA installation:

   ```bash
   export CUDA_HOME=/path/to/your/cuda

   ```

   For example:

   ```bash
   export CUDA_HOME=/usr/local/cuda-11.8

   ```

2. **Ensure CUDA is Installed and Updated**

   Verify that CUDA version 11 or higher is installed on your system. If it is not, please install it according to the official NVIDIA documentation.

By following these steps, your machine should be able to locate and use the CUDA driver, allowing you to proceed with your work.



💥**IMPORTANT**💥 CryoSamba accepts as input (.mrc, .rec, .tif) single 3D files or sequences of 2D (.tif) files. For single files, the "data path" must directly reference the files, while for tif sequences the "data path" should reference the folder containing the sequence. For example, use `path/to/sample_data.rec` or `path/to/tif_folder`. Not referencing the input data properly will lead to errors.
