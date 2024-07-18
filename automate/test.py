import logging
import os
import subprocess
import sys
from functools import wraps

import streamlit as st
from training_setup import handle_exceptions

logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    filename="debug_errors_for_environment.log", encoding="utf-8", level=logging.DEBUG
)
logger = logging.getLogger(__name__)


def is_conda_installed() -> bool:
    """Run a subprocess to see if conda is installled or not"""
    return (
        subprocess.run(
            ["command", "-v", "conda"], capture_output=True, text=True, shell=True
        ).returncode
        == 0
    )


def is_env_active(env_name) -> bool:
    """Use conda env list to check active environments"""
    cmd = "conda env list"
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return f"{env_name}" in result.stdout


# def is_conda_installed() -> bool:
#     try:
#         subprocess.run(["conda", "--version"], capture_output=True, check=True)
#         return True
#     except (subprocess.CalledProcessError, FileNotFoundError):
#         return False


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


@handle_exceptions
def setup_conda():
    st.subheader("Conda Installation")
    if is_conda_installed():
        st.write("Conda is already installled.")
    else:
        if sys.platform.startswith("linux") or sys.platform == "darwin":
            st.write("Conda is not installed. Installing conda ....")
            subprocess.run(
                "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh",
                shell=True,
            )
            subprocess.run("chmod +x Miniconda3-latest-Linux-x86_64.sh", shell=True)
            subprocess.run("bash Miniconda3-latest-Linux-x86_64.sh", shell=True)
            subprocess.run("export PATH=~/miniconda3/bin:$PATH", shell=True)
            subprocess.run("source ~/.bashrc", shell=True)
        else:
            run_command(
                "powershell -Command \"(New-Object Net.WebClient).DownloadFile('https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe', 'Miniconda3-latest-Windows-x86_64.exe')\""
            )
            run_command(
                'start /wait "" Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /S /D=%UserProfile%\\Miniconda3'
            )


@handle_exceptions
def setup_environment(env_name):
    st.subheader(f"Setting up Conda Environment: {env_name}")
    cmd = f"conda init && conda activate {env_name}"
    if is_env_active(env_name):
        st.write(f"Environment '{env_name}' exists.")
        subprocess.run(cmd, shell=True)
    else:
        st.write(f"Creating conda environment: {env_name}")
        subprocess.run(f"conda create --name {env_name} python=3.11 -y", shell=True)
        subprocess.run(cmd, shell=True)
        st.success("Environment has been created", icon="✅")
    st.success("**please copy the command below in the terminal.**", icon="✅")
    st.write(
        "Say you downloaded cryosamba in your downloads folder, open a NEW terminal window and run the following commands:"
    )
    st.code("cd downloads/cryosamba/automate")
    cmd = f"conda init && sleep 3 && source ~/.bashrc && conda activate {env_name} && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip install tifffile mrcfile easydict loguru tensorboard streamlit pipreqs cupy-cuda11x"
    st.code(cmd)


@handle_exceptions
def export_env():
    st.subheader("Exporting Conda Environment")
    subprocess.run("conda env export > environment.yml", shell=True)
    subprocess.run("mv environment.yml ../", shell=True)
    st.write("Environment exported and moved to root directory.")


@handle_exceptions
def setup_environment_for_cryosamba() -> None:
    st.title("Cryosamba Setup Interface")
    st.subheader("Welcome to Cryosamba Setup Interface!")

    st.write(
        "Please take some time to read the instructions and in the case of failures refer to the README for the contact information of relevant parties. *Refer to the video for step by step instructions*"
    )
    lst = [
        "|STEP 1| : **Setup Conda** - if already installed, it shows you that its installed",
        "|STEP 2|: **Make an Environment** - Creates an environment and gives you instructions on which commands to copy",
        "|STEP 3|: **OPTIONAL, Export the Environment** - for programmers who want to look at installed packages",
    ]
    for elem in lst:
        st.markdown(elem)

    if st.button("Setup Conda"):
        setup_conda()

    env_name = st.text_input("Enter environment name", "cryosamba")

    if st.button("2) Setup Environment"):
        setup_environment(env_name)

    if st.button("3) (Optional) Export Environment"):
        export_env()
