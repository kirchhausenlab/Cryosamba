import os
import sys
import subprocess
import unittest
import json


def run_bash_script(script_content):
    # Write the script content to a temporary file
    with open("temp_script.sh", "w") as f:
        f.write(script_content)

    # Make the script executable
    os.chmod("temp_script.sh", 0o755)

    # Run the script and capture output
    process = subprocess.Popen(
        ["./temp_script.sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()

    # Clean up the temporary script
    os.remove("temp_script.sh")

    return stdout, stderr, process.returncode


def get_experiment_name(stdout):
    for line in stdout.split("\n"):
        if "folder made" in line:
            return line.split()[0]
    return None


class TestRunCreation(unittest.TestCase):
    def setUp(self):
        # Your Bash script content here
        self.bash_script = """#!/bin/bash
make_folder() {
    while true; do
      input_name=exp-random
      if  [ -z "$input_name" ]; then
        echo "PLEASE ENTER A NAME, experiment name cannot be empty"
      elif [ -d "../../$input_name" ]; then
        echo "Experiment already exists, please choose a different name"
      else
        DEFAULT_NAME=$input_name
        echo "$DEFAULT_NAME folder made"
        break
      fi
    done
}

# Make train and inference folders
generate_train_and_test_paths(){ 
    mkdir -p "../../runs/$DEFAULT_NAME/train"
    mkdir -p "../../runs/$DEFAULT_NAME/inference"
}


generate_config() {
    base_config=$(cat << EOL
    {
        "train_data": {      
            "max_frame_gap": 6,
            "patch_overlap": [
                16,
                16
            ],
            "patch_shape":[
              256,
              256
            ],
            "split_ratio": 0.95,
            "num_workers": 4
        },
        "train": {
            "load_ckpt_path": null,
            "print_freq": 100,
            "save_freq": 1000,
            "val_freq": 1000,
            "warmup_iters": 300,
            "mixed_precision": true,
            "compile": false
        },
        "optimizer": {
            "lr": 2e-4,
            "lr_decay": 0.99995,
            "weight_decay": 0.0001,
            "epsilon": 1e-08,
            "betas": [
                0.9,
                0.999
            ]
        },
        "biflownet": {
            "pyr_dim": 24,
            "pyr_level": 3,
            "corr_radius": 4,
            "kernel_size": 3,
            "warp_type": "soft_splat",
            "padding_mode": "reflect",
            "fix_params": false
        },
        "fusionnet": {
            "num_channels": 16,
            "padding_mode": "reflect",
            "fix_params": false
        }
    }
EOL
)

    data_path="/nfs/datasync4/inacio/data/denoising/cryosamba/rota/train/"
    max_frame_gap=${max_frame_gap:-6}

    num_iters=${num_iters:-200000}

    batch_size=${batch_size:-32}

    config_file="../../runs/$DEFAULT_NAME/train_config.json"

    train_dir="../exp-random/train"
    # Use jq to merge the base config with user inputs
    echo "$base_config" | jq \
        --arg data_path "$data_path" \
        --arg train_dir "$train_dir"\
        --argjson max_frame_gap "$max_frame_gap" \
        --argjson num_iters "$num_iters" \
        --argjson batch_size "$batch_size" \
        '. + {
            "train_dir":$train_dir,
            "data_path": [$data_path],
            "train_data": (.train_data + {
                "max_frame_gap": $max_frame_gap,
                "batch_size": $batch_size
            }),
            "train": (.train + {
                "num_iters": $num_iters
            })
        }' > "$config_file"

    echo "Config file generated at $config_file"
}


main (){
# Generate a folder
RAND_NUM=$((1+$RANDOM %100))
DEFAULT_NAME=TEST_NAME_EXP-$RAND_NUM
make_folder
# make the folders for paths
generate_train_and_test_paths
# Main script execution
generate_config

}
main
"""
        self.stdout, self.stderr, self.return_code = run_bash_script(self.bash_script)
        self.experiment_name = get_experiment_name(self.stdout)

    def test_script_execution(self):
        self.assertEqual(
            self.return_code, 0, f"Script failed with error: {self.stderr}"
        )

    def test_folder_creation(self):
        if self.experiment_name:
            self.assertTrue(os.path.exists(f"../../runs/{self.experiment_name}"))
            self.assertTrue(os.path.exists(f"../../runs/{self.experiment_name}/train"))
            self.assertTrue(
                os.path.exists(f"../../runs/{self.experiment_name}/inference")
            )
        else:
            self.fail("Experiment name not found in script output")

    def test_config_file_creation(self):
        if self.experiment_name:
            config_path = f"../../runs/{self.experiment_name}/train_config.json"
            self.assertTrue(os.path.exists(config_path))

            # Validate JSON structure
            with open(config_path, "r") as f:
                config = json.load(f)

            self.assertIn("train_data", config)
            self.assertIn("train", config)
            self.assertIn("optimizer", config)
            self.assertIn("biflownet", config)
            self.assertIn("fusionnet", config)
        else:
            self.fail("Experiment name not found in script output")


if __name__ == "__main__":
    unittest.main()
