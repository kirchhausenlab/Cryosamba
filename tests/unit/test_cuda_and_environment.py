import os
import subprocess
import sys
import unittest

from logging_config import logger


class TestCUDA_ENV(unittest.TestCase):
    def test_system_type(self):
        """Test if the current system is cuda compatible or not"""
        curr_system = sys.platform.lower() 
        self.assertIsNotNone(curr_system, msg="Checking if the current system is a valid linux, windows or ubuntu machine")
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
            logger.error(f"Error executing command {str(e)}")
            raise
