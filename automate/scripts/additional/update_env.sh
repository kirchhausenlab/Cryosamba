#!/bin/bash

## Update the environment in cases when libraries might be getting old or not
ENV_NAME="cryosamba_env"
YML_PATH="../../environment.yml"
conda deactivate
conda env update --name $ENV_NAME --file $YML_PATH --prune
conda activate $ENV_NAME

echo "$(date): Conda environment has been updated via cron job scheduled daily"