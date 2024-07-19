import logging
import os
import subprocess
from functools import wraps
from typing import List

import streamlit as st
from file_selector import get_dir, list_directories_in_directory
from training_setup import handle_exceptions

logger = logging.getLogger(__name__)
logging.basicConfig(filename="run_inference.log", encoding="utf-8", level=logging.DEBUG)


@handle_exceptions
def select_gpus() -> List[str]:
    st.text("The following GPUs are not in use, select ones you one want to use! ")
    with st.echo():
        command = "nvidia-smi && nvidia-smi --query-gpu=index,utilization.gpu,memory.free,memory.total,memory.used --format=csv"
        res = subprocess.run(command, shell=True, capture_output=True, text=True)
        st.code(res.stdout)
    options = st.multiselect(
        "Select the GPUs here",
        ["0", "1", "2", "3", "4", "5", "6", "7"],
        ["0", "1", "2"],
    )

    st.write("You selected:", options)
    # print(type(options), options)
    return options


@handle_exceptions
def run_experiment(gpus: str, folder_path: str) -> None:
    print(f"{folder_path}")
    cmd = f"CUDA_VISIBLE_DEVICES={gpus} torchrun --standalone --nproc_per_node=$(echo {gpus} | tr ',' '\n' | wc -l) ../inference.py --config ../{folder_path}/inference_config.json"
    st.text(f"Do you want to run the command: {cmd}?")
    selection = st.radio("Type y/n: ", ["y", "n"], index=None)
    if selection == "n":
        st.write("cancelled")
    elif selection == "y":
        st.write(
            "Dear Reader, copy this command onto your terminal or powershell to train the model, and follow the prompts!"
        )
        st.write(
            "Please open up a new terminal on your machine and navigate to the cryosamba/automate folder. Then run this command"
        )
        st.code(cmd)

        st.code(
            f"SAMPLE COMMAND LOOKS LIKE: \n CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py --config configs/your_config_train.json \n: "
        )


@handle_exceptions
def select_experiment() -> None:
    get_dir()
    st.write("Please enter the experiment you want to run: ")
    input_name = st.text_input("Experiment Name", "")
    base_path = f"../{input_name}"
    if st.button("Check folder"):
        if os.path.exists(base_path):
            st.success(f"Folder {base_path} found")
            st.session_state.folder_found = True
            st.session_state.input_name = input_name
        else:
            st.error(f"Folder {base_path} not found")
            st.session_state.folder = False


@handle_exceptions
def select_experiment_and_run() -> None:
    st.header("Welcome to the CryoSamba Training Runner")
    st.write(
        "Please note that you *need a GPU to run cryosamba, if you cannot see a graph or a table or GPUs **AFTER YOU CHOOSE YOUR Experiment**, your machine does not support cryosamba.*"
    )
    st.write(
        "If the table shows a list of 0s, you have compatible hardware but NO GPUs. Please connect to a machine that has GPUs. Instructions to ssh into a machine for cryosamba are as follows:"
    )
    st.code(
        "# To copy cryosamba, make a zip file of cryosamba and run the following. \n scp cryosamba.zip user_name@remote_server.edu:/path/to/store \n ssh user_name@remote_server.edu && cd /path/to/store \n unzip cryosamba.zip \n \
        \n cd cryosamba/automate \n pip install streamlit \n streamlit run main.py"
    )

    if "folder_found" not in st.session_state:
        select_experiment()
    elif st.session_state.folder_found:
        st.write("We will be running training here:")
        options = select_gpus()
        if not options or len(options) == 0:
            st.error("you did not select any options!")
            st.stop()
        run_experiment(",".join(options), st.session_state.input_name)


# def main():
#     select_experiment_and_run()
# if __name__=="__main__":
#     main()
