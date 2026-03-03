 #!/bin/bash
 
 HF_CACHE="/home/compartido/pabloF/cache"

cd ..
python3 es_indexing_dog.py \
    --es_config "config_elastic.yaml" \
    --index "./indexes/dog/dog_2025.json" \
    --hf_cache_dir $HF_CACHE \
    --data_path "/home/compartido/pabloF/data/DOG/final_data/dog_2025_formated.json" \
    --single_file \
    --chunking "overlap" \
    --embedding "BAAI/bge-m3" \
    --transform_dog