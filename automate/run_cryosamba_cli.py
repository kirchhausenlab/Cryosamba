import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import json
import webbrowser
import os
import subprocess
import webbrowser
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import typer
from loguru import logger
from rich import print as rprint
from rich.console import Console

app = typer.Typer()


def select_gpus() -> Optional[Union[List[str], int]]:
    simple_header("GPU Selection")

    rprint(
        f"[yellow]Please note that you need a nvidia GPU to run CryoSamba. If you cannot see GPU information, your machine may not support CryoSamba.[/yellow]"
    )

    if typer.confirm("Do you want to see detailed GPU information?"):
        command1 = "nvidia-smi"
        res = subprocess.run(command1, shell=True, capture_output=True, text=True)
        print("")
        print(res.stdout)

    command2 = "nvidia-smi --query-gpu=index,utilization.gpu,memory.free,memory.total,memory.used --format=csv"
    res2 = subprocess.run(command2, shell=True, capture_output=True, text=True)

    lst_available_gpus = []
    lines = res2.stdout.split("\n")
    for i, line in enumerate(lines):
        if i == 0 or line == "":
            continue
        lst_available_gpus.append(line.split(",")[0])
    select_gpus = []
    while True:
        rprint(
            f"\n[bold]You have these GPUs left available now: [red]{lst_available_gpus}[/red] and have currently selected these GPUs: [green]{select_gpus}[/green][/bold]"
        )
        gpus = typer.prompt("Add a GPU number: (or Enter F to finish selection)")
        if gpus == "F":
            break
        if gpus in lst_available_gpus:
            select_gpus.append(gpus)
            lst_available_gpus.remove(gpus)
        else:
            rprint(f"[red]Invalid choice![/red]")

    print("")
    if len(select_gpus) == 0:
        rprint(f"[red]You didn't select any GPUs[/red]")
        return -1
    else:
        rprint(f"You have selected the following GPUs: [blue]{select_gpus}[/blue]\n")

    return select_gpus


def run_training(gpus: str, folder_path: str) -> None:
    cmd = f"OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES={gpus} torchrun --standalone --nproc_per_node=$(echo {gpus} | tr ',' '\\n' | wc -l) ../train.py --config ../runs/{folder_path}/train_config.json"
    rprint(
        f"[yellow][bold]!!! Training instructions, read before proceeding !!![/bold][/yellow]"
    )
    rprint(
        f"[bold]* You can interrupt training at any time by pressing CTRL + C, and you can resume it later by running CryoSamba again *[/bold]"
    )
    rprint(
        f"[bold]* Training will run until your specified maximum number of iterations is reached. However, you can monitor the training and validation losses and halt training when you think they have converged/stabilized * [/bold]"
    )
    rprint(
        f"[bold]* You can monitor the losses through here, through the .log file in the experiment training folder, or through TensorBoard (see README on how to run it) *[/bold] \n"
    )
    rprint(
        f"[bold]* The output of the training run will be checkpoint files containing the trained model weights. There is no denoised data output at this point yet. You can used the trained model weights to run inference on your data and then get the denoised outputs. *[/bold] \n"
    )
    if typer.confirm("Do you want to start training?"):
        rprint(f"\n[blue]***********************************************[/blue]\n")
        subprocess.run(cmd, shell=True, text=True)
    else:
        rprint(f"[red]Training aborted[/red]")
        return_screen()


def run_inference(gpus: str, folder_path: str) -> None:
    cmd = f"OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES={gpus} torchrun --standalone --nproc_per_node=$(echo {gpus} | tr ',' '\\n' | wc -l) ../inference.py --config ../runs/{folder_path}/inference_config.json"
    rprint(
        f"[yellow][bold]!!! Inference instructions, read before proceeding !!![/bold][/yellow]"
    )
    rprint(
        f"[bold]* You can interrupt inference at any time by pressing CTRL + C, and you can resume it later by running CryoSamba again *[/bold]"
    )
    rprint(
        f"[bold]* You should have previously run a training session on this experiment in order to run inference * [/bold]"
    )
    rprint(
        f"[bold]* The denoised volume will be generated after the final iteration * [/bold] \n"
    )
    if typer.confirm("Do you want to start inference?"):
        rprint(f"\n[blue]***********************************************[/blue]\n")
        subprocess.run(cmd, shell=True, text=True)
    else:
        rprint(f"[red]Inference aborted[/red]")
        return_screen()


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
        rprint(f"[green]Conda is already installed.[/green]")
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
def setup_environment(
    env_name: str = typer.Option("cryosamba", prompt="Enter environment name")
):
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

        cmd = f"conda init && sleep 3 && source ~/.bashrc && conda activate {env_name} && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip install tifffile mrcfile easydict loguru tensorboard streamlit pipreqs cupy-cuda11x typer webbrowser"
        typer.echo(
            f"Say you downloaded cryosamba in your downloads folder, open a NEW terminal window and run the following commands or hit yes to run it here: \n\n{cmd} "
        )
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


def ask_user(prompt: str, default: Any = None) -> Any:
    return typer.prompt(prompt, default=default)


def ask_user_int(prompt: str, min_value: int, max_value: int, default: int) -> int:
    while True:
        try:
            value = int(ask_user(prompt, default))
            if min_value <= value <= max_value:
                return value
            else:
                rprint(
                    f"[red]Please enter a value between [bold]{min_value}[/bold] and [bold]{max_value}[/bold].[/red]"
                )
        except ValueError:
            rprint(f"[red]Please enter a valid integer.[/red]")


def list_tif_files(path):
    files = []
    # List all files and directories in the specified path
    for entry in os.listdir(path):
        # Construct the full path of the entry
        full_path = os.path.join(path, entry)
        # Check if the entry is a file and ends with '.tif'
        if os.path.isfile(full_path) and entry.endswith(".tif"):
            files.append(full_path)
    return files


@app.command()
def generate_experiment(exp_name: str) -> None:

    rprint(f"Setting up new experiment [green]{exp_name}[/green]")

    exp_path = f"../runs/{exp_name}"

    # Common parameters
    train_dir = f"{exp_path}/train"
    inference_dir = f"{exp_path}/inference"

    while True:
        rprint(
            f"[bold]DATA PATH[/bold]: The path to a single (3D) .tif, .mrc or .rec file, or the path to a folder containing a sequence of (2D) .tif files, ordered alphanumerically matching the Z-stack order. You can use the full path or a path relative from this script's folder."
        )
        data_path = ask_user(
            "Enter your data path",
            f"",
        )
        if not os.path.exists(data_path):
            rprint(f"[red]Data path is invalid. Try again.[/red]")
        else:
            if os.path.isfile(data_path):
                extension = os.path.splitext(data_path)[1]
                if extension not in [".mrc", ".rec", ".tif", ".tiff"]:
                    rprint(
                        f"[red]Extension [bold]{extension}[/bold] is not supported. Try another path.[/red]"
                    )
                else:
                    break
            elif os.path.isdir(data_path):
                files = list_tif_files(data_path)
                if len(files) == 0:
                    rprint(
                        f"[red]Your folder does not contain any tif files. Only sequences of tif files are currently supported. Try another path.[/red]"
                    )
                else:
                    break

    # Training specific parameters
    rprint(
        f"[bold]MAXIMUM FRAME GAP FOR TRAINING[/bold]: explained in the manuscript. We empirically set values of 3, 6 and 10 for data at voxel resolutions of 15.72, 7.86 and 2.62 angstroms, respectively. For different resolutions, try a reasonable value interpolated from the reference ones."
    )
    train_max_frame_gap = ask_user_int("Enter Maximum Frame Gap for Training", 1, 40, 3)
    rprint(
        f"[bold]NUMBER OF ITERATIONS[/bold]: for how many iterations the training session will run. This is an upper limit, and you can halt training before that."
    )
    num_iters = ask_user_int(
        "Enter the number of iterations you want to run", 10000, 200000, 100000
    )
    rprint(
        f"BATCH SIZE: number of data points passed at once to the GPUs. Higher number leads to faster training, but the whole batch might not fit into your GPU's memory, leading to out-of-memory errors. If you're getting these, try to decrease the batch size until they disappear. This number should be a multiple of two."
    )
    batch_size = ask_user_int("Enter the batch size", 2, 256, 8)
    # Inference specific parameters
    rprint(
        f"[bold]MAXIMUM FRAME GAP FOR INFERENCE[/bold]: explained in the manuscript. We recommend using twice the value used for training."
    )
    inference_max_frame_gap = ask_user_int(
        "Enter Maximum Frame Gap for Inference", 1, 80, train_max_frame_gap * 2
    )
    rprint(
        f"[bold]TTA[/bold]: whether to use Test-Time Augmentation or not (see manuscript) during inference."
    )
    tta = typer.confirm(
        "Enable Test Time Augmentation (TTA) for inference?", default=False
    )

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
            "num_workers": 4,
        },
        "train": {
            "num_iters": num_iters,
            "load_ckpt_path": None,
            "print_freq": 100,
            "save_freq": 1000,
            "val_freq": 1000,
            "warmup_iters": 300,
            "mixed_precision": True,
            "compile": False,
        },
        "optimizer": {
            "lr": 2e-4,
            "lr_decay": 0.99995,
            "weight_decay": 0.0001,
            "epsilon": 1e-08,
            "betas": [0.9, 0.999],
        },
        "biflownet": {
            "pyr_dim": 24,
            "pyr_level": 3,
            "corr_radius": 4,
            "kernel_size": 3,
            "warp_type": "soft_splat",
            "padding_mode": "reflect",
            "fix_params": False,
        },
        "fusionnet": {
            "num_channels": 16,
            "padding_mode": "reflect",
            "fix_params": False,
        },
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
            "num_workers": 4,
        },
        "inference": {
            "output_format": "same",
            "load_ckpt_name": None,
            "pyr_level": 3,
            "mixed_precision": True,
            "TTA": tta,
            "compile": False,
        },
    }

    os.makedirs(f"../runs/{exp_name}", exist_ok=True)
    # os.makedirs(f"../runs/{exp_name}/train", exist_ok=True)
    # os.makedirs(f"../runs/{exp_name}/inference", exist_ok=True)

    # Save configs to files
    with open(f"{exp_path}/train_config.json", "w") as f:
        json.dump(train_config, f, indent=4)

    with open(f"{exp_path}/inference_config.json", "w") as f:
        json.dump(inference_config, f, indent=4)

    simple_header(f"Experiment [green]{exp_name}[/green] created")
    return_screen()


def return_screen() -> None:
    if typer.confirm("Return to main menu?"):
        main_menu()
    else:
        exit_screen()


def exit_screen() -> None:
    rprint("[bold]Thank you for using CryoSamba. Goodbye![/bold]")
    quit()


def title_screen() -> None:
    rprint("")
    rprint(
        "[green] ██████╗██████╗ ██╗   ██╗ ██████╗[/green] [yellow]███████╗ █████╗ ███╗   ███╗██████╗  █████╗[/yellow]"
    )
    rprint(
        "[green]██╔════╝██╔══██╗╚██╗ ██╔╝██╔═══██╗[/green][yellow]██╔════╝██╔══██╗████╗ ████║██╔══██╗██╔══██╗[/yellow]"
    )
    rprint(
        "[green]██║     ██████╔╝ ╚████╔╝ ██║   ██║[/green][yellow]███████╗███████║██╔████╔██║██████╔╝███████║[/yellow]"
    )
    rprint(
        "[green]██║     ██╔══██╗  ╚██╔╝  ██║   ██║[/green][yellow]╚════██║██╔══██║██║╚██╔╝██║██╔══██╗██╔══██║[/yellow]"
    )
    rprint(
        "[green]╚██████╗██║  ██║   ██║   ╚██████╔╝[/green][yellow]███████║██║  ██║██║ ╚═╝ ██║██████╔╝██║  ██║[/yellow]"
    )
    rprint(
        "[green] ╚═════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ [/green][yellow]╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝ ╚═╝  ╚═╝[/yellow]"
    )
    rprint("")
    rprint("[bold]Welcome to CryoSamba [white]v1.0[/white] [/bold]")
    rprint(
        "[bold]by Kirchhausen Lab [blue](https://kirchhausen.hms.harvard.edu/)[/blue][/bold]"
    )
    print("")
    typer.echo(
        "Please read the instructions carefully. If you experience any issues reach out to Jose Costa-Filho at joseinacio@tklab.hms.harvard.edu or Arkash Jain at arkash@tklab.hms.harvard.edu"
    )


@app.command()
def main():

    title_screen()

    main_menu()


def main_menu() -> None:

    rprint(f"\n[bold]*** MAIN MENU ***[/bold]\n")

    steps = [
        f"[bold]|1| Set up a new experiment[/bold]",
        f"[bold]|2| Run training[/bold]",
        f"[bold]|3| Run inference[/bold]",
        f"[bold]|4| Exit[/bold]",
    ]

    for step in steps:
        rprint(step)

    print("")
    while True:
        input_cmd = typer.prompt("Choose an option [1/2/3/4]")
        if input_cmd == "1":
            setup_experiment()
            break
        elif input_cmd == "2":
            run_cryosamba("Training")
            break
        elif input_cmd == "3":
            run_cryosamba("Inference")
            break
        elif input_cmd == "4":
            exit_screen()
            break
        else:
            rprint("[red]Invalid option. Please choose either 1, 2, 3 or 4.[/red]")


def simple_header(message) -> None:
    rprint(f"\n[bold]*** {message} ***[/bold]\n")


def setup_cryosamba() -> None:
    simple_header("CryoSamba Setup")

    if typer.confirm("Do you want to setup Conda?"):
        setup_conda()

    if typer.confirm("Do you want to setup the environment?"):
        env_name = typer.prompt("Enter environment name", default="cryosamba")
        setup_environment(env_name)

    if typer.confirm("Do you want to export the environment? (Optional)"):
        export_env()

    rprint("[green]CryoSamba setup finished[/green]")
    return_screen()


def setup_experiment() -> None:
    simple_header("New Experiment Setup")

    if not os.path.exists("../runs"):
        os.makedirs("../runs")
    path = "../runs"

    exp_parent_dir = os.path.join(os.path.dirname(os.getcwd()), "runs")
    rprint(f"Your experiments are stored at [bold]{exp_parent_dir}[/bold]")
    exp_list = list_non_hidden_files(path)
    if len(exp_list) == 0:
        rprint(f"You have no existing experiments.")
    else:
        rprint(f"You have the following experiments: [bold]{exp_list}[/bold]")

    if typer.confirm("Do you want to create a new experiment?"):
        while True:
            exp_name = typer.prompt("Please enter the experiment name")
            exp_path = f"../runs/{exp_name}"
            if os.path.exists(exp_path):
                rprint(
                    f"[red]Experiment [bold]{exp_name}[/bold] already exists. Please choose a new name.[/red]"
                )
            else:
                break
        generate_experiment(exp_name)
    else:
        rprint(f"[yellow]No experiment was created[/yellow]")
        return_screen()

    return exp_name


def list_non_hidden_files(path):
    non_hidden_files = [file for file in os.listdir(path) if not file.startswith(".")]
    return non_hidden_files


def run_cryosamba(mode) -> None:
    simple_header(f"CryoSamba {mode}")

    if not os.path.exists("../runs"):
        os.makedirs("../runs")
    path = "../runs"

    exp_parent_dir = os.path.join(os.path.dirname(os.getcwd()), "runs")
    rprint(f"Your experiments are stored at [bold]{exp_parent_dir}[/bold]")
    exp_list = list_non_hidden_files(path)
    if len(exp_list) == 0:
        rprint(
            f"[red]You have no existing experiments. Set up a new experiment via the main menu.[/red]"
        )
        return_screen()
    else:
        rprint(f"You have the following experiments: [bold]{exp_list}[/bold]")

    while True:
        exp_name = typer.prompt("Please enter the experiment name")
        exp_path = f"../runs/{exp_name}"
        if os.path.exists(exp_path):
            rprint(f"* Experiment [green]{exp_name}[/green] selected *")
            break
        else:
            rprint(
                f"[red]Experiment [bold]{exp_name}[/bold] not found. Please check the experiment name and try again.[/red]"
            )

    selected_gpus = select_gpus()
    if selected_gpus == -1:
        return_screen()

    if mode == "Training":
        run_training(",".join(selected_gpus), exp_name)
    elif mode == "Inference":
        run_inference(",".join(selected_gpus), exp_name)


if __name__ == "__main__":
    typer.run(main)
