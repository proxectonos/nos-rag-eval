#!/bin/bash

source /home/compartido/pabloF/load_env.sh

BASE_DIR="/home/compartido/pabloF/nos-rag-eval/results"
RESULTS_FOLDER="" #/*/
# Recorre todos los subdirectorios dentro de BASE_DIR
for EXP_DIR in "$BASE_DIR"/*/; do
    echo "Procesando directorio: $EXP_DIR"

    # Nombre del fichero de salida dentro del subdirectorio
    OUTFILE="$EXP_DIR/traditional_metric_results.jsonl"

    # Ejecuta el script de Python en modo carpeta
    cd ir-metrics/
    python3 evaluate_ir_metrics.py --folder "$EXP_DIR" --output "$OUTFILE"
    cd ..
done
