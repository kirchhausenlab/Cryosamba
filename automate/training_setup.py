import json
import logging
import os
from functools import wraps
from random import randint

import streamlit as st
from file_selector import get_dir, list_directories_in_directory

logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    filename="debug_errors_for_training.log", encoding="utf-8", level=logging.DEBUG
)
logger = logging.getLogger(__name__)


def handle_exceptions(input_func):
    """Decorator for handling exceptions"""

    @wraps(input_func)
    def wrapper(*args, **kwargs):
        try:
            return input_func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Error in {input_func.__name__}: {str(e)}| Code: {input_func.__code__} "
            )
            st.error(f"An error occurred: {str(e)}")
            raise RuntimeError(
                f"The function {input_func.__name__} failed with the error {str(e)} | Code: {input_func.__code__}"
            )

    return wrapper


@handle_exceptions
def make_folder():
    get_dir()
    st.subheader("Experiment Folder Creation")
    st.write("Enter the name for your experiment:")

    input_name = st.text_input("Experiment Name", "")

    if st.button("Create Experiment Folder"):
        if input_name:
            DEFAULT_NAME = input_name
        else:
            DEFAULT_NAME = f"TEST_NAME_EXP-{randint(1, 100)}"

        st.write(f"Creating experiment folders for '{DEFAULT_NAME}'...")
        try:
            base_path = f"../{DEFAULT_NAME}"
            os.makedirs(f"{base_path}/train", exist_ok=True)
            os.makedirs(f"{base_path}/inference", exist_ok=True)
            st.success(f"Experiment folders created successfully.")
            st.session_state.DEFAULT_NAME = DEFAULT_NAME
            st.session_state.step = "mandatory_params"
        except Exception as e:
            st.error(f"Error creating experiment folders: {str(e)}")


@handle_exceptions
def generate_mandatory_params():
    get_dir()
    st.subheader("Generate JSON Config")
    st.write("Enter the mandatory details: ")

    with st.container():
        st.markdown("**Training Directory Path**")
        train_dir = st.text_input(
            "train_dir", "/nfs/datasync4/inacio/data/denoising/cryosamba/rota/train/"
        )
        st.markdown(
            "_The name of the folder where the checkpoints will be saved (exp-name/train)._"
        )

        st.markdown("**Data Path**")
        data_path = st.text_input(
            "data_path",
            "/nfs/datasync4/inacio/data/raw_data/cryo/novareconstructions/rotacell_grid1_TS09_ctf_3xBin.rec",
        )
        st.markdown(
            "_Filename (for single 3D file) or folder (for 2D sequence) where the raw data is located. This can be a list, in case you want to train on several volumes at the same time._"
        )

        st.markdown("**Maximum Frame Gap**")
        max_frame_gap = st.slider(
            "train_data.max_frame_gap", min_value=1, max_value=20, value=6
        )
        st.markdown(
            "_The maximum frame gap used for training. Explained in the manuscript._"
        )

    if st.button("Next: Add Additional Parameters"):
        st.session_state.mandatory_params = {
            "train_dir": train_dir,
            "data_path": data_path,
            "max_frame_gap": max_frame_gap,
        }
        st.session_state.step = "additional_params"
    elif st.button("Submit"):
        st.session_state.mandatory_params = {
            "train_dir": train_dir,
            "data_path": data_path,
            "max_frame_gap": max_frame_gap,
        }
        st.session_state.step = "generate_config"


@handle_exceptions
def generate_additional_params():
    st.subheader("Change Additional Parameters")
    st.markdown("_Select the section you want to update:_")

    with st.container():
        st.button(
            "Train Data Parameters",
            on_click=lambda: st.session_state.update(
                {"additional_params_section": "train_data"}
            ),
        )
        st.button(
            "Train Parameters",
            on_click=lambda: st.session_state.update(
                {"additional_params_section": "train"}
            ),
        )
        st.button(
            "Optimizer Parameters",
            on_click=lambda: st.session_state.update(
                {"additional_params_section": "optimizer"}
            ),
        )
        st.button(
            "Biflownet Parameters",
            on_click=lambda: st.session_state.update(
                {"additional_params_section": "biflownet"}
            ),
        )
        st.button(
            "Fusionnet Parameters",
            on_click=lambda: st.session_state.update(
                {"additional_params_section": "fusionnet"}
            ),
        )

    additional_params_section = st.session_state.get("additional_params_section", "")

    if additional_params_section == "train_data":
        st.write("**Train Data Parameters**")
        patch_shape_x = st.slider("Patch Shape X", value=256, step=32, max_value=1024)
        patch_shape_y = st.slider("Patch Shape Y", value=256, step=32, max_value=1024)
        patch_overlap_x = st.number_input("Patch Overlap X", min_value=16)
        patch_overlap_y = st.number_input("Patch Overlap Y", min_value=16)
        split_ratio = st.number_input("Split Ratio", value=0.95)
        batch_size = st.number_input("Batch Size", value=32, step=16)
        num_workers = st.number_input("Number of Workers", value=4)
        st.markdown(
            "_X and Y resolution of the patches the model will be trained on. Doesn't need to be square (resolution x = resolution y), but it has to be a multiple of 32._"
        )
        st.markdown(
            "_Number of data points loaded into the GPU at once. Increasing it makes the model train faster (with diminishing returns for large batch size), but requires more GPU memory. Play with it to avoid GPU out of memory errors._"
        )
        if st.button("Save Train Data Parameters"):
            st.session_state.train_data_params = {
                "patch_shape": [patch_shape_x, patch_shape_y],
                "patch_overlap": [patch_overlap_x, patch_overlap_y],
                "split_ratio": split_ratio,
                "batch_size": batch_size,
                "num_workers": num_workers,
            }
            st.success("Train Data Parameters saved")

    if additional_params_section == "train":
        st.write("**Train Parameters**")
        print_freq = st.number_input("Print Frequency", value=100)
        save_freq = st.number_input("Save Frequency", value=1000)
        val_freq = st.number_input("Val Frequency", value=1000)
        num_iters = st.number_input("Number of Iterations", value=200000)
        warmup_iters = st.number_input("Warmup Iterations", value=300)
        compile = st.radio("Compile", [True, False])
        st.markdown(
            "_If true, uses torch.compile for faster training, which is good but takes some minutes to start running the script and it’s somewhat buggy. Recommend using false until you’re comfortable with the code._"
        )
        st.markdown(
            "_Length of the training run. The default value (200k) is a very long time, but you can halt the training whenever you feel it’s fine (see Tensorboard below)._"
        )
        if st.button("Save Train Parameters"):
            st.session_state.train_params = {
                "print_freq": print_freq,
                "save_freq": save_freq,
                "val_freq": val_freq,
                "num_iters": num_iters,
                "warmup_iters": warmup_iters,
                "compile": compile,
            }
            st.success("Train Parameters saved")

    if additional_params_section == "optimizer":
        st.write("**Optimizer Parameters**")
        lr = st.number_input("Learning Rate", value=2e-4, format="%.6f")
        lr_decay = st.number_input("Learning Rate Decay", value=0.99995, format="%.8f")
        weight_decay = st.number_input("Weight Decay", value=0.0001, format="%.8f")
        epsilon = st.number_input("Epsilon", value=1e-8, format="%.10f")
        beta1 = st.number_input("Beta 1", value=0.9, format="%.5f")
        beta2 = st.number_input("Beta 2", value=0.999, format="%.5f")
        if st.button("Save Optimizer Parameters"):
            st.session_state.optimizer_params = {
                "lr": lr,
                "lr_decay": lr_decay,
                "weight_decay": weight_decay,
                "epsilon": epsilon,
                "betas": [beta1, beta2],
            }
            st.success("Optimizer Parameters saved")

    if additional_params_section == "biflownet":
        st.write("**Biflownet Parameters**")
        pyr_dim = st.number_input("Pyr Dimension", value=24)
        pyr_level = st.number_input("Pyr Level", value=3)
        corr_radius = st.number_input("Correlation Radius", value=4)
        kernel_size = st.number_input("Kernel Size", value=3)
        fix_params = st.radio("Fix Params", [True, False])
        if st.button("Save Biflownet Parameters"):
            st.session_state.biflownet_params = {
                "pyr_dim": pyr_dim,
                "pyr_level": pyr_level,
                "corr_radius": corr_radius,
                "kernel_size": kernel_size,
                "fix_params": fix_params,
            }
            st.success("Biflownet Parameters saved")

    if additional_params_section == "fusionnet":
        st.write("**Fusionnet Parameters**")
        num_channels = st.number_input("Number of Channels", value=16)
        if st.button("Save Fusionnet Parameters"):
            st.session_state.fusionnet_params = {"num_channels": num_channels}
            st.success("Fusionnet Parameters saved")

    if st.button("Generate Config"):
        st.session_state.step = "generate_config"


@handle_exceptions
def generate_config():
    DEFAULT_NAME = st.session_state.DEFAULT_NAME
    mandatory_params = st.session_state.mandatory_params

    # Setting default values
    train_data_defaults = {
        "patch_shape": [256, 256],
        "patch_overlap": [16, 16],
        "split_ratio": 0.95,
        "batch_size": 32,
        "num_workers": 4,
    }

    train_defaults = {
        "print_freq": 100,
        "save_freq": 1000,
        "val_freq": 1000,
        "num_iters": 200000,
        "warmup_iters": 300,
        "compile": False,
    }

    optimizer_defaults = {
        "lr": 2e-4,
        "lr_decay": 0.99995,
        "weight_decay": 0.0001,
        "epsilon": 1e-8,
        "betas": [0.9, 0.999],
    }

    biflownet_defaults = {
        "pyr_dim": 24,
        "pyr_level": 3,
        "corr_radius": 4,
        "kernel_size": 3,
        "fix_params": False,
    }

    fusionnet_defaults = {"num_channels": 16}

    train_data_params = {
        **train_data_defaults,
        **st.session_state.get("train_data_params", {}),
    }
    train_params = {**train_defaults, **st.session_state.get("train_params", {})}
    optimizer_params = {
        **optimizer_defaults,
        **st.session_state.get("optimizer_params", {}),
    }
    biflownet_params = {
        **biflownet_defaults,
        **st.session_state.get("biflownet_params", {}),
    }
    fusionnet_params = {
        **fusionnet_defaults,
        **st.session_state.get("fusionnet_params", {}),
    }

    base_config = {
        "train_dir": mandatory_params["train_dir"],
        "data_path": [mandatory_params["data_path"]],
        "train_data": {
            "max_frame_gap": mandatory_params["max_frame_gap"],
            **train_data_params,
        },
        "train": {**train_params, "load_ckpt_path": None, "mixed_precision": True},
        "optimizer": {**optimizer_params},
        "biflownet": {
            **biflownet_params,
            "warp_type": "soft_splat",
            "padding_mode": "reflect",
        },
        "fusionnet": {
            **fusionnet_params,
            "padding_mode": "reflect",
            "fix_params": False,
        },
    }

    config_file = f"../{DEFAULT_NAME}/train_config.json"
    with open(config_file, "w") as f:
        json.dump(base_config, f, indent=4)
    st.success(f"Config file generated successfully at {config_file}")
    st.session_state.config_generated = True


@handle_exceptions
def setup_cryosamba_and_training() -> None:
    st.title("Cryosamba Setup Interface")
    st.write(
        "Welcome to the training setup for cryosamba. Here you can set the parameters for your machine learning configuration."
    )
    st.write(
        "*Note that you have to hit a button twice to see results. The first click shows you a preview of what will happen and the next click runs it*"
    )
    if "DEFAULT_NAME" not in st.session_state:
        make_folder()
    else:
        step = st.session_state.get("step", "mandatory_params")

        if step == "mandatory_params":
            generate_mandatory_params()
        elif step == "additional_params":
            generate_additional_params()
        elif step == "generate_config":
            generate_config()
        else:
            st.success(
                "Configuration already generated. No further modifications allowed."
            )
            if st.button("Exit Setup"):
                st.stop()


# def main():
#     setup_cryosamba_and_training()

# if __name__ == "__main__":
#     main()
