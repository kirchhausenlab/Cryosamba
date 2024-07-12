import logging
import os
import subprocess
from functools import wraps

import streamlit as st
from training_setup import handle_exceptions

logging.basicConfig(level=logging.INFO)
logging.basicConfig(filename='debug_errors_for_environment.log', encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def is_conda_installed() -> bool:
    """Run a subprocess to see if conda is installled or not"""
    return subprocess.run(['command', '-v', 'conda'], capture_output=True, text=True, shell=True).returncode == 0

def is_env_active(env_name) -> bool:
    """Use conda env list to check active environments"""
    cmd="conda env list"
    result=subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return f"{env_name}" in result.stdout

@handle_exceptions
def setup_conda():
    st.subheader("Conda Installation")
    if is_conda_installed():
        st.write("Conda is already installled.")
    else:
        st.write("Conda is not installed. Installing conda ....")
        subprocess.run("wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh", shell=True)
        subprocess.run("chmod +x Miniconda3-latest-Linux-x86_64.sh", shell=True)
        subprocess.run("bash Miniconda3-latest-Linux-x86_64.sh", shell=True)
        subprocess.run("export PATH=~/miniconda3/bin:$PATH", shell=True)
        subprocess.run("source ~/.bashrc", shell=True)


@handle_exceptions
def setup_environment(env_name):
    st.subheader(f"Setting up Conda Environment: {env_name}")
    cmd=f"conda init && source ~/.bashrc && conda activate {env_name}"
    if is_env_active(env_name):
        st.write(f"Environment '{env_name}' exists.")
        subprocess.run(cmd, shell=True)
    else:
        st.write(f"Creating conda environment: {env_name}")
        subprocess.run(f"conda create --name {env_name} python=3.11 -y", shell=True)
        subprocess.run(cmd, shell=True)

        # st.write(f"Activating conda environment: {env_name}")
        # subprocess.run(f"conda activate {env_name}", shell=True)
        
        st.write("Installing PyTorch")
        subprocess.run("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", shell=True)
        
        st.write("Installing other dependencies")
        subprocess.run("pip install tifffile mrcfile easydict loguru tensorboard streamlit pipreqs cupy-cuda11x", shell=True)
        get_reqs="pipreqs ../."
        install_reqs="python3 -m pip install -r ../requirements.txt"
        subprocess.run(get_reqs, shell=True, check=True)
        subprocess.run(install_reqs, shell=True, check=True)

        st.write("Environment setup complete.")

@handle_exceptions
def export_env():
    st.subheader("Exporting Conda Environment")
    subprocess.run("conda env export > environment.yml", shell=True)
    subprocess.run("mv environment.yml ../", shell=True)
    st.write("Environment exported and moved to root directory.")

@handle_exceptions
def setup_environment_for_cryosamba() -> None:
    st.title("Cryosamba Setup Interface")
    st.write("Welcome to Cryosamba Setup Interface!")

    # setup_conda()

    # env_name = "cryosamba-env"
    # setup_environment(env_name)

    # export_env()

    if st.button('Setup Conda'):
        setup_conda() 
        
    env_name = st.text_input("Enter environment name", "incasem")
    
    if st.button('Setup Environment'):
        setup_environment(env_name)
    
    if st.button('Export Environment'):
        export_env()



