#!/bin/bash

HF_CACHE="/home/compartido/pabloF/cache"

cd ..

# Embedding: all-MiniLM
# python es_indexing_press.py \
#     --elastic_config config_elastic.yaml \
#     --index "indexes/press/index_all-minilm-l6-v2_paragraph.json" \
#     --hf_cache_dir $HF_CACHE \
#     --data_path "../datasets/News/Documents" \
#     --chunking "paragraph" \
#     --embedding "sentence-transformers/all-MiniLM-L6-v2"

# Embedding: bge-m3
 python es_indexing_press.py \
    --elastic_config config_elastic.yaml \
    --index "indexes/press/index_bge-m3_paragraph.json" \
    --hf_cache_dir $HF_CACHE  \
    --data_path "../datasets/News/Documents" \
    --chunking "paragraph" \
    --embedding "BAAI/bge-m3"

#Embedding: gemma-300m
# python es_indexing_press.py \
#     --elastic_config config_elastic.yaml \
#     --index "indexes/press/index_gemma-300m_paragraph.json" \
#     --hf_cache_dir $HF_CACHE  \
#     --data_path "../datasets/News/Documents" \
#     --chunking "paragraph" \
#     --embedding "google/embeddinggemma-300m"

#Embedding: granite-english-r2
# python es_indexing_press.py \
#     --elastic_config config_elastic.yaml \
#     --index "indexes/press/index_granite-english-r2_paragraph.json" \
#     --hf_cache_dir $HF_CACHE  \
#     --data_path "../datasets/News/Documents" \
#     --chunking "paragraph" \
#     --embedding "ibm-granite/granite-embedding-english-r2"

#Embedding: qwen3-Embedding-0.6B
# python es_indexing_press.py \
#     --elastic_config config_elastic.yaml \
#     --index "indexes/press/index_qwen3_paragraph.json" \
#     --hf_cache_dir $HF_CACHE  \
#     --data_path "../datasets/News/Documents" \
#     --chunking "paragraph" \
#     --embedding "Qwen/Qwen3-Embedding-0.6B"

#Embedding: None -> BM25 retrieval
# python es_indexing_press.py \
#     --elastic_config config_elastic.yaml \
#     --index "indexes/press/index_no_embedding_paragraph.json" \
#     --hf_cache_dir $HF_CACHE  \
#     --data_path "../datasets/News/Documents" \
#     --chunking "paragraph"
