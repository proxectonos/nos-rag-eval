#!/bin/bash

source /home/compartido/pabloF/load_env.sh

BASE_DIR="/home/compartido/pabloF/nos-rag-eval/results"
CACHE=/home/compartido/pabloF/cache
JUDGE="selene"
DATASET="dog"
REFERENCES=/home/compartido/pabloF/nos-rag-eval/datasets/DOG/Questions/500_tripletas_final.json
# Recorre todos los subdirectorios dentro de BASE_DIR

ARGS=(--dataset "$DATASET" --references "$REFERENCES" --judge_model "$JUDGE" --cache_dir "$CACHE")

for EXP_DIR in "$BASE_DIR"; do
    echo "Procesando directorio: $EXP_DIR"

    # Archivos de salida en el mismo subdirectorio
    OUT_RECALL="$EXP_DIR/judge_recall.jsonl"
    OUT_PRECISION="$EXP_DIR/judge_precision.jsonl"

    cd llm-as-judge/
    # Ejecutar para recall
    echo "  Ejecutando Judge (recall)..."
    python3 judge_evaluator.py "${ARGS[@]}" --folder "$EXP_DIR" --output "$OUT_RECALL" --metric recall

    # Ejecutar para precision
    echo "  Ejecutando Judge (precision)..."
    python3 judge_evaluator.py "${ARGS[@]}" --folder "$EXP_DIR" --output "$OUT_PRECISION" --metric precision
    cd ..
done
