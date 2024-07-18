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
