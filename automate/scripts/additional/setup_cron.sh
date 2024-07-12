#!/bin/bash

## run a cron job every 24 hours to see if dependencies need to be updated or not
CURR_PATH="$(pwd)/update_env.sh"
LOG_FILE="$(pwd)/cron_update.log"

crontabl -l > mycron

echo "0 0 * * * $SCRIPT_PATH >> $LOG_FILE 2>&1" >> mycron

crontab mycron
rm mycron

echo "Cron job has been set up to run $CURR_PATH daily"