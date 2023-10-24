#!/bin/bash

SUMM_PREFIX="summaries/vgpt_"
GT_PATH="data/vgpt_gt.json"

for summ in $SUMM_PREFIX*
do
  basename=$(basename "$summ")
  echo "==========="
  echo "[${basename}]"
  python eval_qefvc/preprocess.py \
    --pred_path "summaries/${basename}" \
    --gt_path "${GT_PATH}" \
    --out_path "data/preprocessed_qefvc/${basename}"\
    --fill_na
done