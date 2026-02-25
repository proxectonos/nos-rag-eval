#!/bin/bash

# Embedding: all-MiniLM
python create_elastic_database_with_embeddings.py \
    --elastic_config config_elastic.yaml \
    --index "indexes/index_all-minilm-l6-v2_paragraph.json" \
    --hf_cache_dir "/home/compartido/pabloF/cache" \
    --data_path "../datasets/News/Documents" \
    --chunking "paragraph" \
    --embedding "sentence-transformers/all-MiniLM-L6-v2"

# Embedding: bge-m3
#  python create_elastic_database_with_embeddings.py \
#     --elastic_config config_elastic.yaml \
#     --index "indexes/index_bge-m3_paragraph.json" \
#     --hf_cache_dir "/home/compartido/pabloF/cache" \
#     --data_path "../datasets/News/Documents" \
#     --chunking "paragraph" \
#     --embedding "BAAI/bge-m3"

#Embedding: gemma-300m
python create_elastic_database_with_embeddings.py \
    --elastic_config config_elastic.yaml \
    --index "indexes/index_gemma-300m_paragraph.json" \
    --hf_cache_dir "/home/compartido/pabloF/cache" \
    --data_path "../datasets/News/Documents" \
    --chunking "paragraph" \
    --embedding "google/embeddinggemma-300m"

#Embedding: granite-english-r2
python create_elastic_database_with_embeddings.py \
    --elastic_config config_elastic.yaml \
    --index "indexes/index_granite-english-r2_paragraph.json" \
    --hf_cache_dir "/home/compartido/pabloF/cache" \
    --data_path "../datasets/News/Documents" \
    --chunking "paragraph" \
    --embedding "ibm-granite/granite-embedding-english-r2"

#Embedding: qwen3-Embedding-0.6B
python create_elastic_database_with_embeddings.py \
    --elastic_config config_elastic.yaml \
    --index "indexes/index_qwen3_paragraph.json" \
    --hf_cache_dir "/home/compartido/pabloF/cache" \
    --data_path "../datasets/News/Documents" \
    --chunking "paragraph" \
    --embedding "Qwen/Qwen3-Embedding-0.6B"

#Embedding: None -> BM25 retrieval
python create_elastic_database_with_embeddings.py \
    --elastic_config config_elastic.yaml \
    --index "indexes/index_no_embedding_paragraph.json" \
    --hf_cache_dir "/home/compartido/pabloF/cache" \
    --data_path "../datasets/News/Documents" \
    --chunking "paragraph"
