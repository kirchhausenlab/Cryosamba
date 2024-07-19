import json
import logging
import os
from functools import wraps
from random import randint

import streamlit as st
from file_selector import get_dir, list_directories_in_directory
from training_setup import handle_exceptions

from logging_config import logger


def folder_exists(folder_path):
    """Check if the given folder path exists."""
    return os.path.exists(folder_path)


@handle_exceptions
def make_folder(is_inference=False):
    get_dir()

    if is_inference:
        st.subheader("Inference Folder Check")
        st.write("Enter the name for your experiment:")
        input_name = st.text_input("Experiment Name", "")
        base_path = f"../runs/{input_name}/inference"
    else:
        st.subheader("Experiment Folder Check")
        st.write("Enter the name for your experiment:")
        input_name = st.text_input("Experiment Name", "")
        base_path = f"../runs/{input_name}/train"

    if st.button("Check Folder"):
        if folder_exists(base_path):
            st.success(f"Folder '{base_path}' found.")
            st.session_state.folder_found = True
            st.session_state.DEFAULT_NAME = input_name
            st.session_state.step = "mandatory_params"
            st.session_state.inference_dir=base_path
        else:
            st.error(f"Folder '{base_path}' not found.")
            st.session_state.folder_found = False


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
            "_The name of the folder where the checkpoints were saved (exp-name/train) in your training run._"
        )

        st.markdown("**Data Path**")
        data_path = st.text_input(
            "data_path",
            "/nfs/datasync4/inacio/data/raw_data/cryo/novareconstructions/rotacell_grid1_TS09_ctf_6xBin.rec",
        )
        st.markdown(
            "_Filename (for single 3D file) or folder (for 2D sequence) where the raw data is located._"
        )

        st.markdown("**Inference Directory Path**")
        inference_dir = st.text_input(
            "inference_dir",
            st.session_state.get('inference_dir', "/nfs/datasync4/inacio/data/denoising/cryosamba/rota/inference/"),
        )
        st.markdown(
            "_The name of the folder where the denoised stack will be saved (exp-name/inference)._"
        )

        st.markdown("**Maximum Frame Gap**")
        max_frame_gap = st.slider(
            "inference_data.max_frame_gap", min_value=1, max_value=40, value=12
        )
        st.markdown(
            "_The maximum frame gap used for inference (usually two times the value used for training). Explained in the manuscript._"
        )

    if st.button("Next: Add Additional Parameters"):
        st.session_state.mandatory_params = {
            "train_dir": train_dir,
            "data_path": data_path,
            "inference_dir": inference_dir,
            "max_frame_gap": max_frame_gap,
        }
        st.session_state.step = "additional_params"
    elif st.button("Submit"):
        st.session_state.mandatory_params = {
            "train_dir": train_dir,
            "data_path": data_path,
            "inference_dir": inference_dir,
            "max_frame_gap": max_frame_gap,
        }
        st.session_state.step = "generate_config"


@handle_exceptions
def generate_additional_params():
    st.subheader("Change Additional Parameters")
    st.markdown("_Select the section you want to update:_")

    with st.container():
        st.button(
            "Inference Data Parameters",
            on_click=lambda: st.session_state.update(
                {"additional_params_section": "inference_data"}
            ),
        )
        st.button(
            "Inference Parameters",
            on_click=lambda: st.session_state.update(
                {"additional_params_section": "inference"}
            ),
        )

    additional_params_section = st.session_state.get("additional_params_section", "")

    if additional_params_section == "inference_data":
        st.write("**Inference Data Parameters**")
        patch_shape_x = st.slider("Patch Shape X", value=256, step=32, max_value=1024)
        patch_shape_y = st.slider("Patch Shape Y", value=256, step=32, max_value=1024)
        patch_overlap_x = st.number_input("Patch Overlap X", min_value=16)
        patch_overlap_y = st.number_input("Patch Overlap Y", min_value=16)
        batch_size = st.number_input("Batch Size", value=32, step=16)
        num_workers = st.number_input("Number of Workers", value=4)
        st.markdown(
            "_X and Y resolution of the patches the model will be trained on. Doesn't need to be square (resolution x = resolution y), but it has to be a multiple of 32._"
        )
        st.markdown(
            "_Number of data points loaded into the GPU at once. Increasing it makes the model train faster (with diminishing returns for large batch size), but requires more GPU memory. Play with it to avoid GPU out of memory errors._"
        )
        if st.button("Save Inference Data Parameters"):
            st.session_state.inference_data_params = {
                "patch_shape": [patch_shape_x, patch_shape_y],
                "patch_overlap": [patch_overlap_x, patch_overlap_y],
                "batch_size": batch_size,
                "num_workers": num_workers,
            }
            st.success("Inference Data Parameters saved")

    if additional_params_section == "inference":
        st.write("**Inference Parameters**")
        output_format = st.radio("Output Format", ["same", "different"])
        load_ckpt_name = st.text_input("Load Checkpoint Name", "")
        pyr_level = st.number_input("Pyr Level", value=3)
        TTA = st.radio("Test-Time Augmentation (TTA)", [True, False])
        compile = st.radio("Compile", [True, False])
        st.markdown(
            "_If true uses test-time augmentation, which makes results slightly better but takes much longer (around 3x) to run._"
        )
        st.markdown(
            "_If true, uses torch.compile for faster training, which is good but takes some minutes to start running the script and it’s somewhat buggy. Recommend using false until you’re comfortable with the code._"
        )
        if st.button("Save Inference Parameters"):
            st.session_state.inference_params = {
                "output_format": output_format,
                "load_ckpt_name": load_ckpt_name,
                "pyr_level": pyr_level,
                "TTA": TTA,
                "compile": compile,
            }
            st.success("Inference Parameters saved")

    if st.button("Generate Config"):
        st.session_state.step = "generate_config"


@handle_exceptions
def generate_config():
    DEFAULT_NAME = st.session_state.DEFAULT_NAME
    mandatory_params = st.session_state.mandatory_params

    # Setting default values
    inference_data_defaults = {
        "patch_shape": [256, 256],
        "patch_overlap": [16, 16],
        "batch_size": 32,
        "num_workers": 4,
    }

    inference_defaults = {
        "output_format": "same",
        "load_ckpt_name": None,
        "pyr_level": 3,
        "TTA": True,
        "mixed_precision": True,
        "compile": True,
    }

    inference_data_params = {
        **inference_data_defaults,
        **st.session_state.get("inference_data_params", {}),
    }
    inference_params = {
        **inference_defaults,
        **st.session_state.get("inference_params", {}),
    }
    base_config = {
        "train_dir": mandatory_params["train_dir"],
        "data_path": [mandatory_params["data_path"]],
        "inference_dir": mandatory_params["inference_dir"],
        "inference_data": {
            "max_frame_gap": mandatory_params["max_frame_gap"],
            **inference_data_params,
        },
        "inference": {**inference_params},
    }

    config_file = f"../runs/{DEFAULT_NAME}/inference_config.json"
    with open(config_file, "w") as f:
        json.dump(base_config, f, indent=4)
    st.success(f"Inference config file generated successfully at {config_file}")
    st.session_state.config_generated = True


@handle_exceptions
def setup_inference() -> None:
    st.title("Cryosamba Inference Setup Interface")
    st.write("Welcome to Cryosamba Inference Setup Interface!")

    if "folder_found" not in st.session_state:
        make_folder(is_inference=True)
    elif st.session_state.folder_found:
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
    else:
        st.error("Please ensure the folder exists before proceeding.")
        make_folder(is_inference=True)

if __name__ == "__main__":
    setup_inference()
