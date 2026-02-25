#!/bin/bash

source /home/compartido/pabloF/load_env.sh

CONFIGS=/home/compartido/pabloF/nos-rag-eval/rag_retriever/configs/experiments
QUESTIONS=/home/compartido/pabloF/nos-rag-eval/datasets/News/Questions/nos-rag-dataset_questions.json

for i in {1..10}; do
    echo "=== Run $i ==="

    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/bm25.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/bge-m3.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/all-minilm-l6-v2.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/qwen3-Embedding.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/gemma-300m.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/granite-english-r2.yaml --run-id $i

    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/bm25_bge-reranker.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/bge-m3_bge-reranker.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/all-minilm-l6-v2_bge-reranker.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/qwen3-Embedding_bge-reranker.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/gemma-300m_bge-reranker.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/granite-english-r2_bge-reranker.yaml --run-id $i

    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/bm25_qwen-reranker.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/bge-m3_qwen-reranker.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/all-minilm-l6-v2_qwen-reranker.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/qwen3-Embedding_qwen-reranker.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/gemma-300m_qwen-reranker.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/granite-english-r2_qwen-reranker.yaml --run-id $i

    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/bm25_jina-v3-reranker.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/bge-m3_jina-v3-reranker.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/all-minilm-l6-v2_jina-v3-reranker.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/qwen3-Embedding_jina-v3-reranker.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/gemma-300m_jina-v3-reranker.yaml --run-id $i
    python3 generate_testset.py --dataset $QUESTIONS --config $CONFIGS/granite-english-r2_jina-v3-reranker.yaml --run-id $i
done