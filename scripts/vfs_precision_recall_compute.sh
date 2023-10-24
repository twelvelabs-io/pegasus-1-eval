#!/bin/bash

# put pred_fp and gold_fp here.
python3 eval_vfs/compute_PR.py --pred_fp vfs_resource/pegasus_vgpt_atomic.json --gold_fp vfs_resource/GT_vgpt_atomic.json --save_fp test_pr.json --gpttype gpt-4