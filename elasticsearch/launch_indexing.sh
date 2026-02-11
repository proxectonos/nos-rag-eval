#!/bin/bash

 python create_elastic_database_with_embeddings.py \
    --elastic_config config_elastic.yaml \
    --index "indexes/index_bge-m3_paragraph.json" \
    --hf_cache_dir "/home/compartido/pabloF/cache" \
    --data_path "../datasets/News" \
    --chunking "paragraph" \
    --embedding "BAAI/bge-m3"
