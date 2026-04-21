#!/bin/bash

source /home/compartido/pabloF/load_env.sh

#CONFIGS=/home/compartido/pabloF/nos-rag-eval/rag_backend/configs/experiments
CONFIGS=/home/compartido/pabloF/nos-rag-eval/experiments/paper_experiments.yaml
QUESTIONS=/home/compartido/pabloF/nos-rag-eval/datasets/DOG/Questions/500_tripletas_final.json

for i in {2..10}; do
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS --run-id $i
done
