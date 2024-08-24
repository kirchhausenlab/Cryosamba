#!/bin/bash

check_conda() {
    STR="Conda installation not found"
    if command -v conda &> /dev/null; then
        STR="Conda is already installed"
        echo $STR
    fi

    if [ "$STR" = "Conda installation not found" ]; then
        echo "Installing conda, please hit yes and enter to install (this may take 3-4 minutes)"
        # Get anaconda
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        chmod +x Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh
        # evaluating conda
        export PATH=~/miniconda3/bin:$PATH 
        rm Miniconda3-latest-Linux-x86_64.*
        source ~/.bashrc
        echo "Conda successfully installed"
    fi
}

# Function to create and set up the conda environment
setup_environment() {
    env_name="cryosamba "

    # Check if the environment already exists
    if conda env list | grep -q "^$env_name\s"; then
        echo "Conda environment '$env_name' already exists. Skipping creation."
    else
        echo "Creating conda environment: $env_name"
        conda create --name $env_name python=3.11 -y
    fi
}

activate_env() {
    env_name="cryosamba"
    echo "Activating conda environment: $env_name"
    source ~/miniconda3/bin/activate $env_name
    
    echo "Installing PyTorch"
    pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
    
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
echo "*** CRYOSAMBA INSTALLATION ***"
echo "* Installing Conda"
check_conda
echo "* Creating the CryoSamba environment *"
setup_environment

echo "* Installing required libraries *"
activate_env
echo "* Exporting environment file *"
export_env
}
main
