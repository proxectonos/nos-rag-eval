#!/bin/bash

source /home/compartido/pabloF/load_env.sh

BASE_DIR="/home/compartido/pabloF/nos-rag-eval/results/generations_retrievals"
CACHE=/home/compartido/pabloF/cache
JUDGE="selene"
DATASET="press"
REFERENCES=/home/compartido/pabloF/nos-rag-eval/datasets/News/Questions/nos-rag-dataset_questions.json
#DATASET="dog"
#REFERENCES=/home/compartido/pabloF/nos-rag-eval/datasets/DOG/Questions/500_tripletas_final.json
# Recorre todos los subdirectorios dentro de BASE_DIR

ARGS=(--dataset "$DATASET" --references "$REFERENCES" --judge_model "$JUDGE" --cache_dir "$CACHE")

for EXP_DIR in "$BASE_DIR"; do
    echo "Procesando directorio: $EXP_DIR"

    # Archivos de salida en el mismo subdirectorio
    OUT_RECALL="$EXP_DIR/judge_recall.jsonl"
    OUT_PRECISION="$EXP_DIR/judge_precision.jsonl"
    OUT_FAITHFULNESS="$EXP_DIR/judge_faithfulness.jsonl"

    cd llm-as-judge/
    # Ejecutar para recall
    # echo "  Ejecutando Judge (recall)..."
    # python3 judge_evaluator.py "${ARGS[@]}" --folder "$EXP_DIR" --output "$OUT_RECALL" --metric recall

    # Ejecutar para precision
    # echo "  Ejecutando Judge (precision)..."
    # python3 judge_evaluator.py "${ARGS[@]}" --folder "$EXP_DIR" --output "$OUT_PRECISION" --metric precision
    
    # Ejecutar para precision
    echo "  Ejecutando Judge (faithfulness)..."
    python3 judge_evaluator.py "${ARGS[@]}" --folder "$EXP_DIR" --output "$OUT_FAITHFULNESS" --metric faithfulness
    
    cd ..
done
