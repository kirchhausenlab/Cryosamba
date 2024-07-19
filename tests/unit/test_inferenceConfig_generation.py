import json
import os
import subprocess
import unittest
from pathlib import Path

from logging_config import logger


class TestInferenceConfig(unittest.TestCase):
    def setUp(self):
        curr_path = Path(__name__).resolve().parent
        path_to_dir = curr_path.parent.parent
        self.path_to_inference_file = path_to_dir / "random_exp" / "inference_config.json"
        self.path_to_experiment= path_to_dir / "random_exp"

    def test_verify_path(self):
        try:
            check_path = self.path_to_experiment.exists()
            self.assertTrue(check_path, msg=f"Experiment folder with inference config not found")

            check_train_folder = (self.path_to_experiment / "train").exists()
            self.assertTrue(check_train_folder, msg="Train folder not found")
            

            check_inference_folder = (self.path_to_experiment / "inference").exists()
            self.assertTrue(check_inference_folder, msg="Inference folder not found")

            check_inference_json = self.path_to_inference_file.exists()
            self.assertTrue(check_inference_json, msg="Inference json does not exist")
        except Exception as e:
            logger.error("❌ Error checking folder: %s", str(e))
            self.fail(f"Folder existence checks failed: {str(e)}")

    def test_valid_json_format(self):
        try:
            with open(self.path_to_inference_file, 'r') as INFERENCE_FILE:
                config=json.load(INFERENCE_FILE)
            self.assertIsInstance(config, dict, "Config file is not a valid json format")
        except json.JSONDecodeError as e:
            logger.error("❌ error not a valid json: %s", str(e))
            logger.critical("❌ error not a valid json: %s", str(e))
            self.fail(f"config file is not a valid Json {str(e)}")
        except Exception as e:
            logger.error("❌ error checking json format: %s", str(e))
            self.fail(f"error checking json format {str(e)}")

    def test_read_write_access(self):
        try:
            self.assertTrue(os.access(self.path_to_inference_file), os.R_OK), "config file is not readable")
            self.assertTrue(os.access(self.path_to_inference_file), os.W_OK), "config file is not writable")
        except Exception as e:
            logger.error("❌ error checking json format: %s", str(e))
            self.fail(f"Error reading and writing the JSON", str(e))

    def test_verify_config(self):
        try:
            with open(self.path_to_inference_file, 'r') as f:
                config = json.load(f)
            
            # Check for mandatory parameters
            self.assertIn('train_dir', config, "train_dir is missing from config")
            self.assertIn('data_path', config, "data_path is missing from config")
            self.assertIn('inference_dir', config, "inference_dir is missing from config")
            self.assertIn('inference_data', config, "inference_data section is missing from config")
            self.assertIn('max_frame_gap', config['inference_data'], "max_frame_gap is missing from inference_data")

            # Check for additional parameters
            self.assertIn('inference', config, "inference section is missing from config")

            # Check specific values
            self.assertIsInstance(config['data_path'], list, "data_path should be a list")
            self.assertGreater(len(config['data_path']), 0, "data_path list is empty")
            
            # Check inference_data parameters
            self.assertEqual(config['inference_data']['patch_shape'], [256, 256], "Incorrect default patch_shape")
            self.assertEqual(config['inference_data']['patch_overlap'], [16, 16], "Incorrect default patch_overlap")
            self.assertEqual(config['inference_data']['batch_size'], 32, "Incorrect default batch_size")
            self.assertEqual(config['inference_data']['num_workers'], 4, "Incorrect default num_workers")

            # Check inference parameters
            self.assertIn('output_format', config['inference'], "output_format is missing from inference section")
            self.assertIn('load_ckpt_name', config['inference'], "load_ckpt_name is missing from inference section")
            self.assertIn('pyr_level', config['inference'], "pyr_level is missing from inference section")
            self.assertIn('TTA', config['inference'], "TTA is missing from inference section")
            self.assertIn('mixed_precision', config['inference'], "mixed_precision is missing from inference section")
            self.assertIn('compile', config['inference'], "compile is missing from inference section")

            # Check default values for inference
            self.assertEqual(config['inference']['pyr_level'], 3, "Incorrect default pyr_level")
            self.assertTrue(config['inference']['TTA'], "Incorrect default TTA value")
            self.assertTrue(config['inference']['mixed_precision'], "Incorrect default mixed_precision value")
            self.assertTrue(config['inference']['compile'], "Incorrect default compile value")

        except Exception as e:
            logger.error("❌ Error verifying inference config: %s", str(e))
            self.fail(f"Inference config verification failed: {str(e)}")


