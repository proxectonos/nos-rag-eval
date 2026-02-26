#!/bin/bash

# Loop through all json files starting with retrieved_dataset
for file in results/retrieved_dataset_*.json; do
    # Extract the model names (fields 3 and 4 when splitting by _)
    # filename: retrieved_dataset_bge-m3_bge_run1.json
    model1=$(echo "$file" | cut -d'_' -f3)
    model2=$(echo "$file" | cut -d'_' -f4)
    
    # Construct folder name
    folder="${model1}_${model2}"
    
    # Create folder and move file
    mkdir -p "results/$folder"
    mv "$file" "results/$folder/"
done