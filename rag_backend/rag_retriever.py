from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from utils.ConfigLoader import ConfigLoader, ExperimentConfig
from rag_backend.retriever.Reranker import Reranker
from rag_backend.retriever.Retriever import Retriever
import torch
from typing import Dict, Union
from enum import Enum
import pprint


class RAG:
    def __init__(self, config: Union[str, ExperimentConfig]):
        """
        Initialize the Retriever and Reranker mode.
        """
        # Initialize retriever
        print("Initializing retriever...")
        if isinstance(config, str):
            self.config = ConfigLoader.load(config)
        else:
            self.config = config
        self.elastic_config = ConfigLoader.load_elastic(self.config.database.elastic_config_file)
        self.retriever = self.__initialize_retriever()
        print("RAG system initialized successfully.")
    
    def __initialize_retriever(self):
        strategy_name = self.config.retriever.retrieval_strategy
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
            vector = embedding_model.encode(search_query).tolist()
            return {
                "knn": {
                    "field": "text_embedding",
                    "query_vector": vector,
                }
            }
        
        initial_search_url = F"{self.elastic_config.endpoint}/{self.config.database.elastic_index}/_search?size={self.config.retriever.num_docs_retrieval}"

        reranker = Reranker(
            model_name=self.config.reranker.reranker_model,
            hf_cache_dir=self.config.general_config.hf_cache_dir,
            use_fp16=True,  # Use half-precision for efficiency
            normalize=True  # Normalize scores to 0-1 range
        ) if self.config.reranker.use_reranking else None

        return Retriever(
            search_url = initial_search_url,
            search_func = bm25_query if not self.config.retriever.embedding_model else vector_query,
            es_home = getattr(self.elastic_config, "home", None),
            es_user = getattr(self.elastic_config, "username", None),
            es_password = getattr(self.elastic_config, "password", None),
            es_endpoint = getattr(self.elastic_config, "endpoint", None),
            reranker = reranker,
            num_docs_retrieval = self.config.retriever.num_docs_retrieval,
            num_docs_reranker = self.config.retriever.num_docs_reranker
        )

    def retrieve_contexts(self, user_query: str):        
        # Retrieve relevant documents
        reranked_docs = self.retriever.invoke(user_query, self.config.database.elastic_index)
        #print(reranked_docs)
        # Store source information
        reranked_docs_structured = []
        for i, (doc, score) in enumerate(reranked_docs):
            # Get document content and metadata
            #print(doc)
            score = doc.get('metadata', {}).get('score', None)
            source_data = {
                "id": i+1,
                "score": score,
                "content": doc.get('text'),
                "metadata": doc.get('metadata',{})
            }
            reranked_docs_structured.append(source_data)
        
        return reranked_docs_structured