#!/bin/bash

select_experiment_location(){

echo "Enter the name of the experiment you want to run inference for:"
read -r EXP_NAME

if [ ! -d "../../runs/$EXP_NAME" ]; then
echo "Experiment does not exist, please make one! You can run setup_experiment.sh to do so"
exit 1
fi

}

generate_config() {
    base_config=$(cat << EOL
    {   
        "inference_dir": "/$EXP_NAME/inference",
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

train_dir=./$EXP_NAME/train

echo "Enter the data path "
read -r data_path
while true; do
  if [ -z "$data_path" = "" ]; then
    echo "Please enter a valid data path"
    data_path=${data_path:-""}
  else
    break
  fi
done


echo "Enter the max frame gap for the inference(usually 2 x value use for training, press Enter for default of 12)"
read -r max_frame_gap
max_frame_gap=${max_frame_gap:-12}

echo "Enter TTA (uses test-time augmentation for slightly better results however is 3x slower to run), press enter for default:true)"
read -r TTA
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
main
