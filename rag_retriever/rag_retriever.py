import os
import sys
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import torch
from configs.ConfigLoader import ConfigLoader
from retriever.Reranker import Reranker
from retriever.Retriever import Retriever
from langchain_elasticsearch import ElasticsearchStore, ElasticsearchRetriever
from typing import Dict
from enum import Enum
import pprint


class ElasticSearchStrategy(Enum):
    BM25 = ElasticsearchStore.BM25RetrievalStrategy()
    APROX = ElasticsearchStore.ApproxRetrievalStrategy()
    EXACT = ElasticsearchStore.ExactRetrievalStrategy()
    SPARSE = ElasticsearchStore.SparseVectorRetrievalStrategy()
    SIMILARITY = "SIMILARITY"

class RAG:
    def __init__(self, config_file):
        """
        Initialize the Retriever and Reranker mode.
        """
        # Initialize retriever
        print("Initializing retriever...")
        self.config = ConfigLoader.load(config_file)
        self.elastic_config = ConfigLoader.load_elastic(self.config.database.elastic_config_file)
        self.retriever = self.__initialize_retriever()
        print("RAG system initialized successfully.")
    
    def __initialize_retriever(self):
        strategy_name = self.config.retriever.retrieval_strategy
        retrieval_strategy = ElasticSearchStrategy[strategy_name].value
        if self.config.retriever.embedding_model:
            embedding_model = SentenceTransformer(self.config.retriever.embedding_model, 
                                                  cache_folder=self.config.general_config.hf_cache_dir)
            embedding_model.to('cuda' if torch.cuda.is_available() else 'cpu')

        #https://python.langchain.com/docs/integrations/retrievers/elasticsearch_retriever/
        def bm25_query(search_query: str) -> Dict:
            return {
                "query": {
                    "match": {
                        "text": search_query,
                    },
                },
            }
        def vector_query(search_query: str) -> Dict:
            vector = embedding_model.encode(search_query)
            return {
                "knn": {
                    "field": "text_embedding",
                    "query_vector": vector,
                }
            }
        es_client = Elasticsearch(
            hosts=[self.elastic_config.endpoint],
            basic_auth=(self.elastic_config.username, self.elastic_config.password)
        )
        vectorstore_retriever = ElasticsearchRetriever(
            es_client=es_client,
            index_name=self.config.database.elastic_index,
            content_field="text",
            body_func=bm25_query if strategy_name == "BM25" else vector_query,
        )
        reranker = Reranker(
            model_name=self.config.reranker.reranker_model,
            hf_cache_dir=self.config.general_config.hf_cache_dir,
            use_fp16=True,  # Use half-precision for efficiency
            normalize=True  # Normalize scores to 0-1 range
        ) if self.config.reranker.use_reranking else None

        return Retriever(
            vectorstore=vectorstore_retriever,
            top_k=self.config.retriever.query_top_k,
            reranker=reranker,
            initial_retrieve_count=self.config.retriever.initial_retrieve_count
        )

    def retrieve_contexts(self, user_query: str):        
        # Retrieve relevant documents
        initial_docs, final_docs = self.retriever.invoke(user_query)
        # Format context from retrieved and reranked documents in Galician
        context = "\n\n".join([f"Documento {i+1}: {doc.page_content}" for i, (doc,_) in enumerate(final_docs)])
        
        # Store source information
        source_info = []
        initial_docs_info = []
        for i, (doc,score) in enumerate(final_docs):
            # Get document content and metadata
            source_data = {
                "id": i+1,
                "score": score,
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, "metadata") else {}
            }
            source_info.append(source_data)
        for i, (doc,score) in enumerate(initial_docs):
            # Get document content and metadata
            source_data = {
                "id": i+1,
                "score": score,
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, "metadata") else {}
            }
            initial_docs_info.append(source_data)

        return source_info, initial_docs_info