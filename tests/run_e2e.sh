#!/bin/bash
#The test_sample folder has some sample data produced by running 500 iterations of cryosamba on a DGX A100 GPU with a batch size of 
# 16 and max frame gap of 3. The following test recreates that scenario for users and will take around 15 minutes to run (both train and inference)
# Once you have the training and inference done, navigate to the test_rotacell folder, where you will find our sample results. Compare
# what you get by running this test to our sample as a sanity check for cryosamba's validity
#
# Download the ndc10gfp_g7_l1_ts_002_ctf_6xBin.rec file from dropbox and put it in the cryosamba folder before running the tests

echo "Please make sure you change the paths for the data in the test_sample folder"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py --config test_sample/train_config.json
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 inference.py --config test_sample/inference_config.json
