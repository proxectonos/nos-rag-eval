#!/bin/bash

source /home/compartido/pabloF/load_env.sh

CONFIGS=/home/compartido/pabloF/nos-rag-eval/rag_backend/configs/experiments
QUESTIONS=/home/compartido/pabloF/nos-rag-eval/datasets/News/Questions/nos-rag-dataset_questions.json

python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/bm25.yaml --run-id 1
