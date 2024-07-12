#!/bin/bash

select_gpus(){
    echo "The following GPU's are currently not in use"
    nvidia-smi && nvidia-smi --query-gpu=index,utilization.gpu,memory.free,memory.total,memory.used --format=csv
    echo "--"
    echo ""
    echo "Enter which GPU you want (seperate using commas, e.g. - 2,3)"
    read -r gpu_indices
}

select_experiment() {
    echo "Enter the name of the experiment you want to run:"
    read -r EXP_NAME

    if [ ! -d "../../../$EXP_NAME" ]; then
    echo "Experiment does not exist, please make one! You can run setup_experiment.sh to do so"
    exit 1
    fi

    if [ ! -f "../../../$EXP_NAME/inference_config.json" ]; then
    echo "../../../$EXP_NAME/config.json"
    echo "config does not exist, please make one! You can run setup_experiment.sh to do so"
    exit 1
    fi
}

command_construct(){
# Construct the command
cmd="CUDA_VISIBLE_DEVICES=$gpu_indices torchrun --standalone --nproc_per_node=$(echo $gpu_indices | tr ',' '\n' | wc -l) ../../../train.py --config ../../../$EXP_NAME/inference_config.json"
echo "Do you want to run the command $cmd?"
echo "--"

echo "Type y/n:"
read -r selection
if [ $selection = "n" ]; then
echo "cancelled!!"
else
    echo "running on GPUs, $cmd"
    eval "$cmd"
fi
}


main() {
    # Select the GPUs
    select_gpus
    # Select experiment to run
    select_experiment
    # Eval command
    command_construct
}

main
