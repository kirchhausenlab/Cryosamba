#!/bin/bash


make_folder() {
    echo "What do you want to name your experiment, type below: "
    read -r input_name
    if ! [ -z "$input_name" ]; then
    DEFAULT_NAME=$input_name
    # echo $DEFAULT_NAME
    fi
}

# Make train and inference folders
generate_train_and_test_paths(){ 
    mkdir -p "../../../$DEFAULT_NAME/train"
    mkdir -p "../../../$DEFAULT_NAME/inference"
}


# Function to generate JSON config
generate_config() {
    base_config=$(cat << EOL
    { 
        "train_data": {      
            "max_frame_gap": 6,
            "patch_overlap": [
                16,
                16
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
            "mixed_precision": true
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

echo "Enter the training directory path (press Enter for default):"
read -r train_dir
train_dir=${train_dir:-"/nfs/datasync4/inacio/data/denoising/cryosamba/rota/train/"}

echo "Enter the data path (press Enter for default):"
read -r data_path
data_path=${data_path:-"/nfs/datasync4/inacio/data/raw_data/cryo/novareconstructions/rotacell_grid1_TS09_ctf_3xBin.rec"}

echo "Enter the maximum frame gap (press Enter for default: 6):"
read -r max_frame_gap
max_frame_gap=${max_frame_gap:-6}

echo "Enter the number of iterations (press Enter for default: 200000):"
read -r num_iters
num_iters=${num_iters:-200000}

echo "Enter the batch size (press Enter for default: 32):"
read -r batch_size
batch_size=${batch_size:-32}

config_file="../../../$DEFAULT_NAME/config.json"

# Use jq to merge the base config with user inputs
echo "$base_config" | jq \
    --arg train_dir "$train_dir" \
    --arg data_path "$data_path" \
    --argjson max_frame_gap "$max_frame_gap" \
    --argjson num_iters "$num_iters" \
    --argjson batch_size "$batch_size" \
    '. + {
        "train_dir": $train_dir,
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