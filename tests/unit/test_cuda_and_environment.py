import os
import subprocess
import sys
import unittest

from logging_config import logger


class TestCUDA_ENV(unittest.TestCase):
    def test_system_type(self):
        """Test if the current system is cuda compatible or not"""
        curr_system = sys.platform.lower()
        self.assertIsNotNone(
            curr_system,
            msg="Checking if the current system is a valid linux, windows or ubuntu machine",
        )
        self.assertNotEqual(curr_system, "darwin")

    def check_cuda(self):
        if sys.platform.startswith("linux"):
            self.run_command("sudo apt-get update && sudo apt-get install -y cuda")

    def run_command(self, command, shell=True):
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
    
    def check_conda_installation(self):
        try:
            output, err = self.run_command(["conda", "--version"])
            if output == float("inf"):
                self.assertIsNone("Command failed to run, conda not found %s", str(err))
            if output or not err:
                self.assertTrue(f"Conda exists - {output}")
        except FileNotFoundError as e:
            self.assertFalse(f"Conda was not found! {str(e)}")
            logger.critical("❌ Conda not found %s", str(e))

    def check_active_env(self, env_name="cryosamba"):
        try:
            cmd = "conda env list"
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            active_env = False
            if f"{env_name}" in result.stdout:
                active_env=True
                self.assertTrue(active_env, msg=f"Environment exists! {env_name}")
            else:
                self.assertFalse(active_env, msg=f"no activate environment aside from base")
        except Exception as e:
            # check if conda is installed or not
            logger.critical("❌ could not find environment %s", str(e))

    def check_installed_packages(self, env_name="cryosamba"):
        try:
            output, _ = self.run_command(f"conda activate {env_name} && conda list")
            packages_lst=["torch", "torchvision", "torchaudio", "tifffile", "mrcfile", "easydict", "loguru", "streamlit", "cupy-cuda11x"]
        except Exception as e:
            logger.error(f"💀 Error executing command {str(e)}")



