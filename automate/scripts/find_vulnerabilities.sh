#!/bin/bash

install_bandit_and_run_tests(){
	conda activate cryosamba_env
	conda install bandit
	conda install bandit[toml]
	bandit -r ../. -ll
}

install_bandit_and_run_tests
