 #!/bin/bash
 
HF_CACHE="/home/compartido/pabloF/cache"

cd ..
# bge-M3
# python3 es_indexing_dog.py \
#     --es_config "config_elastic.yaml" \
#     --index "./indexes/dog/dog_2025.json" \
#     --hf_cache_dir $HF_CACHE \
#     --data_path "/home/compartido/pabloF/data/DOG/final_data/dog_2025_formated.json" \
#     --single_file \
#     --chunking "overlap" \
#     --embedding "BAAI/bge-m3" \
#     --transform_dog
# bm25
python3 es_indexing_dog.py \
    --es_config "config_elastic.yaml" \
    --index "./indexes/dog/dog_2025_bm25.json" \
    --hf_cache_dir $HF_CACHE \
    --data_path "/home/compartido/pabloF/data/DOG/final_data/dog_2025_formated.json" \
    --single_file \
    --chunking "overlap" \
    --transform_dog
# gemma-300M
python3 es_indexing_dog.py \
    --es_config "config_elastic.yaml" \
    --index "./indexes/dog/dog_2025_gemma.json" \
    --hf_cache_dir $HF_CACHE \
    --data_path "/home/compartido/pabloF/data/DOG/final_data/dog_2025_formated.json" \
    --single_file \
    --chunking "overlap" \
    --embedding "google/embeddinggemma-300m" \
    --transform_dog

#Qwen-3-0.6B
python3 es_indexing_dog.py \
    --es_config "config_elastic.yaml" \
    --index "./indexes/dog/dog_2025_qwen3.json" \
    --hf_cache_dir $HF_CACHE \
    --data_path "/home/compartido/pabloF/data/DOG/final_data/dog_2025_formated.json" \
    --single_file \
    --chunking "overlap" \
    --embedding "Qwen/Qwen3-Embedding-0.6B" \
    --transform_dog

python3 es_indexing_dog.py \
    --es_config "config_elastic.yaml" \
    --index "./indexes/dog/dog_2025_qwen3.json" \
    --hf_cache_dir $HF_CACHE \
    --data_path "/home/compartido/pabloF/data/DOG/final_data/dog_2025_formated.json" \
    --single_file \
    --chunking "overlap" \
    --embedding "Qwen/Qwen3-Embedding-0.6B" \
    --transform_dog