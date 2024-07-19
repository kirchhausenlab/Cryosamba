import json
import os
import unittest
from pathlib import Path

from logging_config import logger


class TestTrainConfig(unittest.TestCase):
    def setUp(self):
        self.folder_name = "random_exp"
        self.curr_path = Path(__file__).resolve().parent
        self.path_to_experiments = self.curr_path.parent.parent
        self.config_path = self.path_to_experiments / self.folder_name / "train_config.json"

    def test_generate_test_config(self):
        try: 
            self.assertTrue(self.config_path.exists(), "Config file was not generated")
        except Exception as e:
            logger.error("❌ Error checking config file: %s", str(e))
            self.fail(f"Config file check failed: {str(e)}")

    def test_verify_config(self):
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Check for mandatory parameters
            self.assertIn('train_dir', config, "train_dir is missing from config")
            self.assertIn('data_path', config, "data_path is missing from config")
            self.assertIn('train_data', config, "train_data section is missing from config")
            self.assertIn('max_frame_gap', config['train_data'], "max_frame_gap is missing from config")

            # Check for additional parameters
            self.assertIn('train', config, "train section is missing from config")
            self.assertIn('optimizer', config, "optimizer section is missing from config")
            self.assertIn('biflownet', config, "biflownet section is missing from config")
            self.assertIn('fusionnet', config, "fusionnet section is missing from config")

            # Check specific values
            self.assertEqual(config['train_dir'], "/nfs/datasync4/inacio/data/denoising/cryosamba/rota/train/", "Incorrect train_dir")
            self.assertEqual(config['data_path'], ["/nfs/datasync4/inacio/data/raw_data/cryo/novareconstructions/rotacell_grid1_TS09_ctf_3xBin.rec"], "Incorrect data_path")
            self.assertEqual(config['train_data']['max_frame_gap'], 6, "Incorrect max_frame_gap")
            self.assertEqual(config['train_data']['patch_shape'], [256, 256], "Incorrect patch_shape")
            self.assertEqual(config['train_data']['patch_overlap'], [16, 16], "Incorrect patch_overlap")
            self.assertEqual(config['train_data']['split_ratio'], 0.95, "Incorrect split_ratio")
            self.assertEqual(config['train_data']['batch_size'], 32, "Incorrect batch_size")
            self.assertEqual(config['train_data']['num_workers'], 4, "Incorrect num_workers")
            self.assertEqual(config['train']['num_iters'], 200000, "Incorrect num_iters")
            self.assertEqual(config['train']['compile'], False, "Incorrect compile value")
            self.assertAlmostEqual(config['optimizer']['lr'], 0.0002, places=6, msg="Incorrect learning rate")
            self.assertEqual(config['biflownet']['pyr_dim'], 24, "Incorrect pyr_dim")
            self.assertEqual(config['fusionnet']['num_channels'], 16, "Incorrect num_channels")

        except Exception as e:
            logger.error("❌ Error verifying config: %s", str(e))
            self.fail(f"Config verification failed: {str(e)}")

    def test_check_folder_created(self):
        try:
            check_path = (self.path_to_experiments / self.folder_name).exists()
            self.assertTrue(check_path, msg=f"Experiment folder '{self.folder_name}' not found")

            train_folder = self.path_to_experiments / self.folder_name / "train"
            inference_folder = self.path_to_experiments / self.folder_name / "inference"
            self.assertTrue(train_folder.exists(), msg="Train folder not found")
            self.assertTrue(inference_folder.exists(), msg="Inference folder not found")
        except Exception as e:
            logger.error("❌ Error checking folder creation: %s", str(e))
            self.fail(f"Folder creation check failed: {str(e)}")

    def test_config_file_permissions(self):
        try:
            self.assertTrue(os.access(self.config_path, os.R_OK), "Config file is not readable")
            self.assertTrue(os.access(self.config_path, os.W_OK), "Config file is not writable")
        except Exception as e:
            logger.error("❌ Error checking config file permissions: %s", str(e))
            self.fail(f"Config file permissions check failed: {str(e)}")

    def test_config_file_format(self):
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            self.assertIsInstance(config, dict, "Config file is not a valid JSON dictionary")
        except json.JSONDecodeError as e:
            logger.error("❌ Error parsing JSON: %s", str(e))
            self.fail(f"Config file is not a valid JSON: {str(e)}")
        except Exception as e:
            logger.error("❌ Error checking config file format: %s", str(e))
            self.fail(f"Config file format check failed: {str(e)}")

if __name__ == "__main__":
    unittest.main()
