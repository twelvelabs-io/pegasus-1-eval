#!/bin/bash

# Define common arguments for all scripts
PRED_DIR="data/preprocessed_qefvc"
OUTPUT_DIR="results/qefvc"
NUM_TASKS="1"
MODEL_NAME="gpt-4"

## iterate over multiple model results
for fname in $PRED_DIR/*
do
  basename=$(basename "$fname" .json)

  echo "==========="
  echo "[${basename}]"

  # Run the "correctness" evaluation script
  python eval_qefvc/evaluate_benchmark_1_correctness.py \
    --pred_path "${fname}" \
    --output_dir "${OUTPUT_DIR}/${basename}/correctness_eval" \
    --output_json "${OUTPUT_DIR}/${basename}/correctness_results.json" \
    --api_key $OPENAI_API_KEY \
    --num_tasks $NUM_TASKS \
    --model_name $MODEL_NAME

  # Run the "detailed orientation" evaluation script
  python eval_qefvc/evaluate_benchmark_2_detailed_orientation.py \
    --pred_path "${fname}" \
    --output_dir "${OUTPUT_DIR}/${basename}/detailed_eval" \
    --output_json "${OUTPUT_DIR}/${basename}/detailed_orientation_results.json" \
    --api_key $OPENAI_API_KEY \
    --num_tasks $NUM_TASKS \
    --model_name $MODEL_NAME

  # Run the "contextual understanding" evaluation script
  python eval_qefvc/evaluate_benchmark_3_context.py \
    --pred_path "${fname}" \
    --output_dir "${OUTPUT_DIR}/${basename}/context_eval" \
    --output_json "${OUTPUT_DIR}/${basename}/contextual_understanding_results.json" \
    --api_key $OPENAI_API_KEY \
    --num_tasks $NUM_TASKS \
    --model_name $MODEL_NAME

  echo "All evaluations completed!"
  echo "==========="
done