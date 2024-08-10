#!/bin/bash

check_conda() {
    STR="no"
    if command -v conda &> /dev/null; then
    STR="conda is installed no need to install it"
    echo $STR
    fi

    if [ "$STR" = "no" ]; then
        echo "Installing conda, please hit yes and enter to install (this may take 3-4 minutes)"
        # Get anaconda
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        chmod +x Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh
        # evaluating conda
        export PATH=~/miniconda3/bin:$PATH 
        rm Miniconda3-latest-Linux-x86_64.*
        source ~/.bashrc     
    fi
}

# Function to create and set up the conda environment
setup_environment() {
    env_name="cryosamba"
    
    echo "Creating conda environment: $env_name"
    conda create --name $env_name python=3.11 -y
   
    echo "running conda init"
    conda init --all 
   

    sleep 5
    echo "update shell"
    source ~/.bashrc
    }

activate_env() {
    env_name="cryosamba"
    echo "Activating conda environment: $env_name"
    conda activate $env_name
    
    echo "Installing PyTorch"
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    echo "Installing other dependencies"
    pip install tifffile mrcfile easydict loguru tensorboard cupy-cuda11x streamlit typer
    echo "Environment setup complete"

}

export_env(){
    conda env export > environment.yml
    mv environment.yml ../../
}

main(){
# check conda, if it doesnt exist install it
echo "Hello User, we are intialling cryosamba"
check_conda
echo "conda installed"
# install necessary dependencies 
setup_environment

activate_env
# update bash path
source ~/.bashrc     
# make a yml and move it
export_env
}
main
