import json
import os
import subprocess
import unittest

import yaml

from logging_config import logger


class TestTrainConfig(unittest):
    def generate_test_config(self):
        try:
            print("test")
        except Exception as e:
            logger.error("❌ Error executing command: %s", str(e))

        def verify_test_config(self):
            try:
                print("run")
            except Exception as e:
                logger.error("❌ Error executing command: %s ", str(e))


if __name__ == "__main__":
    unittest.main()
