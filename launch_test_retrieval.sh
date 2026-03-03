#!/bin/bash

source /home/compartido/pabloF/load_env.sh

#CONFIGS=/home/compartido/pabloF/nos-rag-eval/rag_backend/configs/experiments
CONFIGS=/home/compartido/pabloF/nos-rag-eval/experiments/test_experiments.yaml
QUESTIONS=/home/compartido/pabloF/nos-rag-eval/datasets/News/Questions/nos-rag-dataset_questions.json

for i in {1..1}; do
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS --run-id $i
done
