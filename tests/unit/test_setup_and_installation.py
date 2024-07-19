import os
import shlex
import subprocess
import sys
import unittest

from logging_config import logger


class TestCUDA_ENV(unittest.TestCase):
    def run_command(self, command, shell=True):
        """
        Run a shell command and return its output and error (if any).
        """
        try:  
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
                    
            output, error = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error executing command: {command}\nError: {error}")
            
            return output, error
        except Exception as e:
            logger.error(f"💀 Error executing command: {str(e)}")
            return None, str(e)

    def test_system_type(self):
        """Test if the current system is compatible with cryosamba"""
        curr_system = sys.platform.lower()
        self.assertIsNotNone(
            curr_system,
            msg="Checking if the current system is a valid linux, windows or ubuntu machine",
        )
        self.assertNotEqual(curr_system, "darwin", msg="System is macOS, which is not CUDA compatible")

    def test_check_cuda(self):
        """Check if cuda is installed or not"""
        if sys.platform.startswith("linux"):
            output, error = self.run_command("nvidia-smi")
            self.assertIsNotNone(output, msg="CUDA is not installed or not functioning properly")
        elif sys.platform.startswith("win"):
            output, error = self.run_command("nvidia-smi")
            self.assertIsNotNone(output, msg="CUDA is not installed or not functioning properly")
        else:
            self.skipTest("This test is only applicable on Linux or Windows systems")

    def test_conda_installation(self):
        output, error = self.run_command("conda --version")
        self.assertIsNotNone(output, msg=f"Conda is not installed or not found in PATH. Error: {error}")
        logger.info("Conda version: %s", output)

    def test_active_env(self, env_name="cryosamba"):
        output, error = self.run_command("conda env list")
        self.assertIn(env_name, output, msg=f"Environment {env_name} does not exist")
        logger.info(f"Environment {env_name} exists")

    def test_installed_packages(self, env_name="cryosamba"):
        packages_lst = ["torch", "torchvision", "torchaudio", "tifffile", "mrcfile", "easydict", "loguru", "streamlit"]
        output, error = self.run_command(f"conda run -n {env_name} conda list", shell=True)
        self.assertIsNotNone(output, msg=f"Failed to list packages in environment {env_name}. Error: {error}")
        for package in packages_lst:
            self.assertIn(package, output, msg=f"Package {package} is not installed in environment {env_name}")
            logger.info(f"✅ {package} is installed")

if __name__ == "__main__":
    unittest.main()
