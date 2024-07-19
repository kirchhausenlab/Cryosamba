import logging
import os
from test import setup_environment_for_cryosamba

import streamlit as st
from inference_setup import setup_inference
from run_inference import select_experiment_and_run
from run_training import select_experiment_and_run_training
from training_setup import setup_cryosamba_and_training

from logging_config import logger


def main():
    st.sidebar.title("Cryosamba Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            "Choose your options!",
            "Setup Environment",
            "Setup Training",
            "Run Training",
            "Setup Inference",
            "Run Inference",
        ],
    )
    if app_mode == "Choose your options!":
        st.header(
            "Please look at the options from the dropdown to either setup, train or run inferences. CAREFUL, IT WILL ONLY RUN ON A WINDOWS OR A LINUX"
        )
    elif app_mode == "Setup Environment":
        setup_environment_for_cryosamba()
    elif app_mode == "Setup Training":
        setup_cryosamba_and_training()
    elif app_mode == "Run Training":
        select_experiment_and_run_training()
    elif app_mode == "Setup Inference":
        setup_inference()
    elif app_mode == "Run Inference":
        select_experiment_and_run()
    st.sidebar.title("Workflow Overview")
    st.sidebar.markdown("## Step-by-Step guide")
    st.sidebar.write("1) Setup Environment")
    st.sidebar.write("2) Setup Training")
    st.sidebar.write("3) Run Training")
    st.sidebar.write("4) Setup Inference")
    st.sidebar.write("5) Run Inference")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
        raise 
