import logging
import os
import subprocess
from functools import wraps

import streamlit as st
from training_setup import handle_exceptions

logging.basicConfig(level=logging.INFO, filename="debug_errors_for_environment.log", encoding="utf-8")
logger = logging.getLogger(__name__)

def run_command(command, shell=True):
    process = subprocess.Popen(command, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output, error = process.communicate()
    if process.returncode != 0:
        st.error(f"Error executing command: {command}\nError: {error}")
        logger.error(f"Error executing command: {command}\nError: {error}")
    return output, error

def is_conda_installed() -> bool:
    return subprocess.run(["conda", "--version"], capture_output=True, text=True).returncode == 0

def is_env_active(env_name) -> bool:
    output, _ = run_command("conda env list")
    return f"{env_name}" in output

@handle_exceptions
def setup_conda():
    st.subheader("Conda Installation")
    if is_conda_installed():
        st.write("Conda is already installed.")
    else:
        st.write("Conda is not installed. Installing conda ....")
        run_command("wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh")
        run_command("chmod +x Miniconda3-latest-Linux-x86_64.sh")
        run_command("bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3")
        run_command("$HOME/miniconda3/bin/conda init bash")
        st.write("Conda installed. Please restart the application for changes to take effect.")

@handle_exceptions
def setup_environment(env_name):
    st.subheader(f"Setting up Conda Environment: {env_name}")
    if is_env_active(env_name):
        st.write(f"Environment '{env_name}' exists.")
    else:
        st.write(f"Creating conda environment: {env_name}")
        run_command(f"conda create --name {env_name} python=3.11 -y")
    
    st.write(f"Activating conda environment: {env_name}")
    run_command(f" pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    run_command(f" pip install tifffile mrcfile easydict loguru tensorboard streamlit pipreqs cupy-cuda11x")
    st.write("Environment setup complete.")

@handle_exceptions
def export_env(env_name):
    st.subheader("Exporting Conda Environment")
    run_command(f"conda env export -n {env_name} > environment.yml")
    run_command("mv environment.yml ../")
    st.write("Environment exported and moved to root directory.")

@handle_exceptions
def setup_environment_for_cryosamba() -> None:
    st.title("Cryosamba Setup Interface")
    st.write("Welcome to Cryosamba Setup Interface!")

    if st.button("Setup Conda"):
        setup_conda()

    env_name = st.text_input("Enter environment name", "cryosamba")
    if st.button("Setup Environment"):
        setup_environment(env_name)

    if st.button("Export Environment"):
        export_env(env_name)

if __name__ == "__main__":
    setup_environment_for_cryosamba()
