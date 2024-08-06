import typer
import json
from typing import Dict, Any
from typing import Optional
import os
import subprocess
from logging_config import logger
from typing import List, Optional
from rich.console import Console
from rich import print as rprint
from functools import wraps
from pathlib import Path
app = typer.Typer()

def select_gpus() -> Optional[List[str]|int]:
    print("The following GPUs are not in use, select ones you one want to use! ")
    command1 = "nvidia-smi"
    command2 = "nvidia-smi --query-gpu=index,utilization.gpu,memory.free,memory.total,memory.used --format=csv"
    res = subprocess.run(command1, shell=True, capture_output=True, text=True)
    print(res.stdout)
    res2 = subprocess.run(command2, shell=True, capture_output=True, text=True)
    print(res2.stdout)
    lst_available_gpus =[] 
    lines = res2.stdout.split('\n')
    for i, line in enumerate(lines):
        if i == 0 or line == '':
            continue
        lst_available_gpus.append(line.split(',')[0])
    select_gpus=[]
    while True:
        rprint(f"\n[bold]You have these cores avaialable now: [red]{lst_available_gpus}[/red] and have currently selected these GPUs: [green]{select_gpus}[/green][/bold]")
        gpus = typer.prompt("Which GPUs would you like to pick: (Enter E to exit)")
        if gpus=="E":
            break
        if gpus in lst_available_gpus:
            select_gpus.append(gpus)
            lst_available_gpus.remove(gpus)
        else:
            typer.echo("Invalid choice!!")
    rprint(f"You have selected the following GPUs: [blue]{select_gpus}[/blue]\n\n")
    if len(select_gpus) == 0:
        return -1
    return select_gpus

def run_experiment(gpus: str, folder_path: str) -> None:
    cmd = f"CUDA_VISIBLE_DEVICES={gpus} torchrun --standalone --nproc_per_node=$(echo {gpus} | tr ',' '\\n' | wc -l) ../train.py --config ../runs/{folder_path}/train_config.json"
    rprint(f"[underline]Do you want to run the command: {cmd}?[/underline]\n\n")
    if typer.confirm("Run the command?"):
        print("Dear Reader, copy this command onto your terminal or powershell to train the model, and follow the prompts!")
        rprint("[italic]SAMPLE COMMAND LOOKS LIKE:")
        rprint("[grey]CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py --config configs/your_config_train.json[/grey] \n\n")
        typer.echo("Copy and run this command for training {cmd}")
        cmd2 = f"CUDA_VISIBLE_DEVICES={gpus} torchrun --standalone --nproc_per_node=$(echo {gpus} | tr ',' '\\n' | wc -l) ../inference.py --config ../runs/{folder_path}/inference_config.json"
        typer.echo("For running an inference run the following command: {cmd2}")
        print(f"Please open up a new terminal on your machine and navigate to the cryosamba/automate folder. Then run this command or you can hit Ctrl-C and run this command: \n")
        if typer.confirm("Do you want to open TensorBoard to monitor the training?"):
            tensorboard_cmd = f"tensorboard --logdir ../runs/{folder_path}/train"
            subprocess.Popen(tensorboard_cmd, shell=True)
            webbrowser.open("http://localhost:6006")
    else:
        print("Command execution cancelled.")

def handle_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            typer.echo(f"An error occurred: {str(e)}")
            logger.exception("An exception occurred")
            raise typer.Exit(code=1)
    return wrapper

@handle_exceptions
def is_conda_installed() -> bool:
    """Run a subprocess to see if conda is installed or not"""
    try:
        subprocess.run(
            ["conda", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False

@handle_exceptions
def is_env_active(env_name) -> bool:
    """Use conda env list to check active environments"""
    cmd = "conda env list"
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return f"{env_name}" in result.stdout

def run_command(command, shell=True):
    process = subprocess.Popen(
        command,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    output, error = process.communicate()
    if process.returncode != 0:
        typer.echo(f"Error executing command: {command}\nError: {error}", err=True)
        logger.error(f"Error executing command: {command}\nError: {error}")
    return output, error

@app.command()
@handle_exceptions
def setup_conda():
    """Setup Conda installation"""
    typer.echo("Conda Installation")
    if is_conda_installed():
        typer.echo("Conda is already installed.")
    else:
        if sys.platform.startswith("linux") or sys.platform == "darwin":
            typer.echo("Conda is not installed. Installing conda ....")
            subprocess.run(
                "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh",
                shell=True,
            )
            subprocess.run("chmod +x Miniconda3-latest-Linux-x86_64.sh", shell=True)
            subprocess.run("bash Miniconda3-latest-Linux-x86_64.sh", shell=True)
            subprocess.run("export PATH=~/miniconda3/bin:$PATH", shell=True)
            subprocess.run("source ~/.bashrc", shell=True)
        else:
            run_command(
                "powershell -Command \"(New-Object Net.WebClient).DownloadFile('https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe', 'Miniconda3-latest-Windows-x86_64.exe')\""
            )
            run_command(
                'start /wait "" Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /S /D=%UserProfile%\\Miniconda3'
            )

@app.command()
@handle_exceptions
def setup_environment(env_name: str = typer.Option("cryosamba", prompt="Enter environment name")):
    """Setup Conda environment"""
    typer.echo(f"Setting up Conda Environment: {env_name}")
    cmd = f"conda init && conda activate {env_name}"
    if is_env_active(env_name):
        typer.echo(f"Environment '{env_name}' exists.")
        subprocess.run(cmd, shell=True)
    else:
        typer.echo(f"Creating conda environment: {env_name}")
        subprocess.run(f"conda create --name {env_name} python=3.11 -y", shell=True)
        subprocess.run(cmd, shell=True)
        typer.echo("Environment has been created")
        typer.echo("**please copy the command below in the terminal.**")

        cmd = f"conda init && sleep 3 && source ~/.bashrc && conda activate {env_name} && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip install tifffile mrcfile easydict loguru tensorboard streamlit pipreqs cupy-cuda11x"
        typer.echo("Say you downloaded cryosamba in your downloads folder, open a NEW terminal window and run the following commands or hit yes to run it here: \n\n{cmd} ")
        run_cmd = typer.prompt("Enter (y/n): ")
        while True:
            if run_cmd == "n":
                typer.echo(cmd)
                break
            elif run_cmd == "y":
                subprocess.run(cmd, shell=True, text=True)
                break 

@app.command()
@handle_exceptions
def export_env():
    """Export Conda environment"""
    typer.echo("Exporting Conda Environment")
    subprocess.run("conda env export > environment.yml", shell=True)
    subprocess.run("mv environment.yml ../", shell=True)
    typer.echo("Environment exported and moved to root directory.")

def make_train_inference_dir() -> None:
    if not os.path.exists("../runs"):
        os.makedirs("../runs")
    paths="../runs"
    rprint(f"You currently have the following experiments {os.listdir(paths)}")
    input_cmd = typer.prompt("Do you want to create a new experiment? Enter (y/n): ")
    while True:
        if input_cmd == "n":
            typer.echo("Exiting!!")
            break
        if input_cmd == "y":
            typer.echo("Generating configs for you, enter your parameters below: ")
            exp_name=typer.prompt("Enter the experiment name")
            exp_path = f"../runs/{exp_name}"
            os.makedirs(f"../runs/{exp_name}/train", exist_ok=True)
            os.makedirs(f"../runs/{exp_name}/inference", exist_ok=True)
            rprint(f"You currently have the following experiments {os.listdir(exp_path)}")
            generate_configs(exp_path)
            break

def ask_user(prompt: str, default: Any = None) -> Any:
    return typer.prompt(prompt, default=default)

def ask_user_int(prompt: str, min_value: int, max_value: int, default: int) -> int:
    while True:
        try:
            value = int(ask_user(prompt, default))
            if min_value <= value <= max_value:
                return value
            else:
                typer.echo(f"Please enter a value between {min_value} and {max_value}.")
        except ValueError:
            typer.echo("Please enter a valid integer.")

@app.command()
def generate_configs(exp_path:str)->None:
    typer.echo("Generate JSON Configs for Training and Inference")

    # Common parameters
    train_dir = f"{exp_path}/train"
    inference_dir=f"{exp_path}/inference"
    print(f"You have the following files outside: {os.listdir('../')}")
    curr_path = Path(__name__).resolve().parent.parent
    data_path = ask_user("Enter Data Path", f"{curr_path}/rotacell_grid1_TS09_ctf_3xBin.rec")
    
    # Training specific parameters
    train_max_frame_gap = ask_user_int("Enter Maximum Frame Gap for Training", 1, 20, 6)
    num_iters = ask_user_int("Enter the number of iterations you want to run", 80000, 200000, 100000)
    batch_size = ask_user_int("Enter the training batch size", 16, 32, 16)
    # Inference specific parameters
    inference_max_frame_gap = ask_user_int("Enter Maximum Frame Gap for Inference", 1, 40, 12)
    tta = typer.confirm("Enable Test Time Augmentation (TTA) for inference?", default=False)

    # Generate training config
    train_config = {
        "train_dir": train_dir,
        "data_path": data_path,
        "train_data": {      
            "max_frame_gap": train_max_frame_gap,
            "patch_overlap": [16, 16],
            "patch_shape": [256, 256],
            "split_ratio": 0.95,
            "batch_size": batch_size,
            "num_workers": 4
        },
        "train": {
            "num_iters": num_iters,
            "load_ckpt_path": None,
            "print_freq": 100,
            "save_freq": 1000,
            "val_freq": 1000,
            "warmup_iters": 300,
            "mixed_precision": True,
            "compile": False
        },
        "optimizer": {
            "lr": 2e-4,
            "lr_decay": 0.99995,
            "weight_decay": 0.0001,
            "epsilon": 1e-08,
            "betas": [0.9, 0.999]
        },
        "biflownet": {
            "pyr_dim": 24,
            "pyr_level": 3,
            "corr_radius": 4,
            "kernel_size": 3,
            "warp_type": "soft_splat",
            "padding_mode": "reflect",
            "fix_params": False
        },
        "fusionnet": {
            "num_channels": 16,
            "padding_mode": "reflect",
            "fix_params": False
        }
    }

    # Generate inference config
    inference_config = {
        "train_dir": train_dir,
        "data_path": data_path,
        "inference_dir": inference_dir,
        "inference_data": {
            "max_frame_gap": inference_max_frame_gap,
            "patch_shape": [256, 256],
            "patch_overlap": [16, 16],
            "batch_size": batch_size,
            "num_workers": 4
        },
        "inference": {
            "output_format": "same",
            "load_ckpt_name": None,
            "pyr_level": 3,
            "mixed_precision": True,
            "tta": tta,
            "compile": False
        }
    }

    # Save configs to files
    with open(f'{exp_path}/train_config.json', 'w') as f:
        json.dump(train_config, f, indent=4)
    
    with open(f'{exp_path}/inference_config.json', 'w') as f:
        json.dump(inference_config, f, indent=4)

    typer.echo("Configs generated and saved as train_config.json and inference_config.json")


@app.command()
def main():
    """Cryosamba Setup Interface"""
    typer.echo("Welcome to Cryosamba Setup Interface!")
    typer.echo("Please take some time to read the instructions and in the case of failures refer to the README for the contact information of relevant parties. *Refer to the video for step by step instructions*")
    
    steps = [
        "|STEP 1| : Setup Conda - if already installed, it shows you that it's installed",
        "|STEP 2|: Make an Environment - Creates an environment and gives you instructions on which commands to copy",
        "|STEP 3|: OPTIONAL, Export the Environment - for programmers who want to look at installed packages",
    ]
    
    for step in steps:
        typer.echo(step)
    
    if typer.confirm("Do you want to setup Conda?"):
        setup_conda()
    
    if typer.confirm("Do you want to setup the environment?"):
        env_name = typer.prompt("Enter environment name", default="cryosamba")
        setup_environment(env_name)
    
    if typer.confirm("Do you want to export the environment? (Optional)"):
        export_env()
    make_train_inference_dir() 
    print("Please note that you need a GPU to run cryosamba. If you cannot see GPU information, your machine may not support cryosamba.")
    
    selected_gpus = select_gpus()
    if not selected_gpus:
        print("You did not select any GPUs. Exiting.")
        return
    path = "../runs/"  
    rprint(f"You have the following experiments: [bold]{os.listdir(path)}[/bold]")
    experiment_name = typer.prompt("Please enter the experiment name")
    base_path = f"../runs/{experiment_name}"
    if os.path.exists(base_path):
        print(f"Folder {base_path} found")
        run_experiment(",".join(selected_gpus), experiment_name)
    else:
        print(f"Folder {base_path} not found. Please check the experiment name and try again.")
if __name__ == "__main__":
    typer.run(main)
