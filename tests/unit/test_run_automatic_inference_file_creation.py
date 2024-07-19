import json
import os
import subprocess
import sys
import unittest


def run_bash_script(script_content):
    with open("temp_script.sh", "w") as f:
        f.write(script_content)
    os.chmod("temp_script.sh", 0o755)
    process = subprocess.Popen(
        [
            "./temp_script.sh",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()

    os.remove("temp_script.sh")
    return stdout, stderr, process.returncode


class TestInferenceRunCreation(unittest.TestCase):
    def setUp(self):
        # Your Bash script content here
        self.bash_script = """#!/bin/bash

        select_experiment_location(){

        EXP_NAME="exp-random"
        if [ ! -d "../../runs/$EXP_NAME" ]; then
        echo "Experiment does not exist, please make one! You can run setup_experiment.sh to do so"
        exit 1
        fi
        }

        generate_config() {
            base_config=$(cat << EOL
            {   
                "inference_dir": "../$EXP_NAME/inference",
                "inference_data": {
                    "patch_shape": [
                        256,
                        256
                    ],
                    "patch_overlap": [
                        16,
                        16
                    ],
                    "batch_size": 32,
                    "num_workers": 4
                },
                "inference": {
                    "output_format": "same",
                    "load_ckpt_name": null,
                    "pyr_level": 3,
                    "mixed_precision": true
                }
            }
        EOL
        )

        train_dir=../$EXP_NAME/train
        data_path=""
        

        max_frame_gap=${max_frame_gap:-12}

        TTA=${TTA:-true}

        compile=false
        config_file="../../runs/$EXP_NAME/inference_config.json"

        # use jq to merge the base config with user inputs
        echo "$base_config" | jq \
            --arg train_dir "$train_dir" \
            --arg data_path "$data_path" \
            --argjson max_frame_gap "$max_frame_gap" \
            --argjson TTA "$TTA" \
            --argjson compile "$compile" \
            '. + {
                "train_dir": $train_dir,
                "data_path": [$data_path],
                "inference_data" : (.inference_data+ {
                    "max_frame_gap": $max_frame_gap,
                }),
                "inference": (.inference+ {
                    "TTA": $TTA,
                    "compile": $compile 
                })
            }' > "$config_file"
        echo "generated config file!"
        }


        main(){
            select_experiment_location
            generate_config
        }
        main"""
        self.stdout, self.stderr, self.return_code = run_bash_script(self.bash_script)
        self.experiment_name = "exp-random"

    def test_script_execution(self):
        self.assertEqual(
            self.return_code, 0, f"Script failed with error: {self.stderr}"
        )

    def test_config_file_creation(self):
        if self.experiment_name:
            config_path = f"../../runs/{self.experiment_name}/inference_config.json"
            self.assertTrue(os.path.exists(config_path))

            # Validate JSON structure
            with open(config_path, "r") as f:
                config = json.load(f)

            self.assertIn("inference", config)
            self.assertIn("inference_dir", config)
            self.assertIn("inference_data", config)
            self.assertIn("train_dir", config)
            self.assertIn("data_path", config)
        else:
            self.fail("Experiment name not found in script output")


if __name__ == "__main__":
    unittest.main()
