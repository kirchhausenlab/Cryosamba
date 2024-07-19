import logging
import os
import subprocess
import sys

import streamlit as st
from training_setup import handle_exceptions

from logging_config import logger

def run_command(command, shell=True):
    process = subprocess.Popen(
        command,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    output, error = process.communicate()
    if process.returncode != 0:
        st.error(f"Error executing command: {command}\nError: {error}")
        logger.error(f"Error executing command: {command}\nError: {error}")
    return output, error


def is_conda_installed() -> bool:
    try:
        subprocess.run(["conda", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_conda():
    st.write("Conda is not installed. Installing conda...")
    if sys.platform.startswith("linux"):
        run_command(
            "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        )
        run_command("chmod +x Miniconda3-latest-Linux-x86_64.sh")
        run_command("bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3")
        run_command("$HOME/miniconda3/bin/conda init bash")
    elif sys.platform == "darwin":
        run_command(
            "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        )
        run_command("chmod +x Miniconda3-latest-MacOSX-x86_64.sh")
        run_command("bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda3")
        run_command("$HOME/miniconda3/bin/conda init bash")
    elif sys.platform == "win32":
        run_command(
            "powershell -Command \"(New-Object Net.WebClient).DownloadFile('https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe', 'Miniconda3-latest-Windows-x86_64.exe')\""
        )
        run_command(
            'start /wait "" Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /S /D=%UserProfile%\\Miniconda3'
        )
    else:
        st.error("Unsupported operating system")
        return False

    st.write(
        "Conda installed. Please restart the application for changes to take effect."
    )
    return True


@handle_exceptions
def setup_conda():
    st.subheader("Conda Installation")
    if is_conda_installed():
        st.write("Conda is already installed.")
        return True
    else:
        if install_conda():
            st.write("Conda installation completed successfully.")
            st.write("Please restart the application for changes to take effect.")
            return True
        else:
            st.error("Failed to install Conda.")
            return False


@handle_exceptions
def setup_environment_for_cryosamba() -> None:
    st.title("Cryosamba Setup Interface")
    st.write("Welcome to Cryosamba Setup Interface!")

    if not is_conda_installed():
        st.warning(
            "Conda is not installed. You need to install Conda before proceeding."
        )
        if st.button("Install Conda"):
            if setup_conda():
                st.success(
                    "Conda installed successfully. Please restart the application."
                )
            else:
                st.error(
                    "Failed to install Conda. Please try again or install manually."
                )
    else:
        st.success("Conda is installed.")
        if st.button("Setup Environment"):
            env_name = st.text_input("Enter environment name", "cryosamba")
            setup_environment(env_name)
        if st.button("Export Environment"):
            env_name = st.text_input("Enter environment name to export", "cryosamba")
            export_env(env_name)


def setup_environment(env_name):
    st.subheader(f"Setting up Conda Environment: {env_name}")
    if is_env_active(env_name):
        st.write(f"Environment '{env_name}' exists.")
    else:
        st.write(f"Creating conda environment: {env_name}")
        run_command(f"conda create --name {env_name} python=3.11 -y")
    st.write(f"Activating conda environment: {env_name}")
    run_command(
        f"conda activate {env_name} && pip3 install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    )
    run_command(
        f"conda activate {env_name} && pip install tifffile mrcfile easydict loguru tensorboard streamlit pipreqs cupy-cuda11x"
    )
    st.write("Environment setup complete.")


def is_env_active(env_name) -> bool:
    output, _ = run_command("conda env list")
    return f"{env_name}" in output


def export_env(env_name):
    st.subheader("Exporting Conda Environment")
    run_command(f"conda env export -n {env_name} > environment.yml")
    run_command("mv environment.yml ../")
    st.write("Environment exported and moved to root directory.")


if __name__ == "__main__":
    setup_environment_for_cryosamba()
