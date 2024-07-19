import os
import shutil
import sys
import unittest

from tests.unit.test_inferenceConfig_generation import TestInferenceConfig
from tests.unit.test_run_automatic_inference_file_creation import (
    TestInferenceRunCreation,
)
from tests.unit.test_run_automatic_train_file_creation import TestRunCreation
from tests.unit.test_setup_and_installation import TestCUDA_ENV
from tests.unit.test_trainConfig_gen import TestTrainConfig


class IntegrationTest(unittest.TestCase):
    def run_and_verify(self, suite):
        res = unittest.TextTestRunner(verbosity=2).run(suite)
        self.assertTrue(res.wasSuccessful())

    def test_1_setup_and_installation(self):
        suite = unittest.TestLoader().loadTestsFromTestCase(TestCUDA_ENV)
        res = unittest.TextTestRunner(verbosity=2).run(suite)
        self.assertTrue(res.wasSuccessful())

    def test_2_train_config_gen(self):
        suite = unittest.TestLoader().loadTestsFromTestCase(TestRunCreation)
        self.run_and_verify(suite)

    def test_3_train_config_validate(self):
        suite = unittest.TestLoader().loadTestsFromTestCase(TestTrainConfig)
        self.run_and_verify(suite)

    def test_5_inference_config_validate(self):
        suite = unittest.TestLoader().loadTestsFromTestCase(TestInferenceRunCreation)
        self.run_and_verify(suite)

    def test_4_inference_config_gen(self):
        suite = unittest.TestLoader().loadTestsFromTestCase(TestInferenceConfig)
        self.run_and_verify(suite)


if __name__ == "__main__":
    unittest.main(verbosity=2)
