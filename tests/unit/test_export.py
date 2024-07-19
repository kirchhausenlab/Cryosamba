import os
import subprocess
import sys
import unittest

from cryosamba.logging_config import logger


def run_command(command, shell=True):
    try:
        process = subprocess.Popen(
            command,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        output, error = process.communicate()
        if process.returncode != 0:
            logger.error(f"Error executing command: {command}\nError: {error}")
        return output, error
    except subprocess.CalledProcessError as e:
        logger.error(f"💀 Error executing command {str(e)}")
        dummy_return = float("inf")
        return dummy_return, e



def check_installed_packages(env_name="cryosamba"):
        try:
            output, _ = run_command(f"conda activate {env_name} && conda list")
            packages_lst=["torch", "torchvision", "torchaudio", "tifffile", "mrcfile", "easydict", "loguru", "streamlit", "cupy-cuda11x"]
            print(output)
        except Exception as e:
            logger.error(f"💀 Error executing command {str(e)}")
check_installed_packages()
