# RAG Backend

A modular Retrieval-Augmented Generation (RAG) backend that combines document retrieval, reranking, and language model inference.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Adding a New Reranker Model](#adding-a-new-reranker-model)
- [Adding a New Retrieval Model](#adding-a-new-retrieval-model)

## Overview

The RAG system works in three main stages:

1. **Retrieval**: Queries an Elasticsearch index using either BM25 (keyword search) or semantic search (dense embeddings) to fetch the top-k candidate documents.

2. **Reranking**: Scores and reorders the retrieved documents using a specialized reranker model to improve relevance.

3. **Generation**: (Optional) Passes the refined context to a language model to generate a final response.

## Architecture

### Core Components

- **`RAG`** (`rag.py`): Main orchestrator that initializes and manages the retrieval, reranking, and LLM components.

- **`Retriever`** (`retriever/Retriever.py`): Handles document search against Elasticsearch using BM25 or vector search, with optional reranking applied.

- **`Reranker`** (`retriever/Reranker.py`): Wrapper around multiple reranker implementations that reranks documents based on query relevance.

- **`LLMHandler`** (`llm_handler.py`): Wrapper for language model inference, handling model loading, prompt formatting, and response generation.

### Reranker Models Supported

The `Reranker` class automatically selects the appropriate backend based on the model name:

- **FlagEmbedding models** (e.g., `BAAI/bge-reranker-v2-m3`): Uses the `FlagReranker` implementation
- **Qwen rerankers** (e.g., `Qwen3-Reranker-0.6B`): Uses the `Qwen3Reranker` implementation
- **Jina rerankers** (e.g., models starting with "jina"): Uses the `JinaReranker` implementation
- **Other models**: Falls back to `SentenceTransformerReranker` for any other model


## Configuration

The RAG system is configured via a YAML config file. Configuration is typically organized in three main sections: **general config**, **embedders**, **rerankers**, and optional **experiments**. See the [experiments](../experiments/) folder for example configuration files. 

### Example Configuration

```yaml
general_config:
  hf_cache_dir: YOUR_HUGGINGFACE_CACHE
  elastic_config_file: YOUR_ELASTICSEARCH_CONFIG
  dataset_name: dog

retriever_defaults:
  retrieval_strategy: SIMILARITY        # Default strategy when embedder doesn't override
  num_docs_retrieval: 10                # Initial documents to retrieve before reranking
  num_docs_reranker: 3                  # Final documents to return

# Define available embedding models/strategies
embedders:
  - name: bm25
    retrieval_strategy: BM25
    elastic_index: "dog_2025_bm25"
  - name: qwen3
    embedding_model: Qwen/Qwen3-Embedding-0.6B
    elastic_index: "dog_2025_qwen3"
  - name: bge-m3
    embedding_model: BAAI/bge-m3
    elastic_index: "dog_2025"
    
# Define available reranker models
rerankers:
  - name: none                                    # No reranking
  - name: qwen
    reranker_model: Qwen/Qwen3-Reranker-0.6B
  - name: jina
    reranker_model: jinaai/jina-reranker-v3
  - name: bge
    reranker_model: BAAI/bge-reranker-v2-m3

llms:
  - name: salamandra
    llm_model: BSC-LT/salamandra-7b-instruct
    quantization: true
    system_prompt : "You are a helpful and impartial assistant..."

# Optional: Define specific embedder x reranker combinations to run
# If omitted, all combinations are tested
experiments:
  - embedder: bm25
    reranker: none
    llm: salamandra
  - embedder: bm25
    reranker: qwen

  - embedder: bge-m3
    reranker: jina
    
  - embedder: bge-m3
    reranker: bge
    llm: salamandra
```

### Configuration Sections

- **general_config**: Global settings like cache directory, Elasticsearch config, dataset name
- **retriever_defaults**: Default retrieval parameters (can be overridden per embedder)
- **embedders**: List of available embedding models and retrieval strategies
  - `name`: Identifier for the embedder
  - `embedding_model`: HuggingFace model ID (omit for BM25)
  - `retrieval_strategy`: `BM25`, `SIMILARITY`, or custom strategy
  - `elastic_index`: Elasticsearch index to query
  
- **rerankers**: List of available reranker models
  - `name`: Identifier for the reranker
  - `reranker_model`: Model ID from HuggingFace (omitted for "none" baseline)

- **llms**: List of available language models (optional)
  - `name`: Identifier for the LLM
  - `llm_model`: Model ID from HuggingFace
  - `quantization`: Boolean to enable 4-bit quantization for efficiency
  - `system_prompt`: System prompt for the model
  
- **experiments** (optional): Define which embedder-reranker-llm combinations to test
  - `embedder`: Name of the embedder to use
  - `reranker`: Name of the reranker to use
  - `llm`: Name of the LLM to use (optional, omit for retrieval-only)
  - If not provided, all combinations are tested

## Adding a New Reranker Model

### 1. Identify the Reranker Type

First, determine which category your reranker falls into:
- **FlagEmbedding-based**: Models from `BAAI` (e.g., `BAAI/bge-reranker-v2-m3`)
- **Qwen-based**: Models from Qwen (name starts with "Qwen")
- **Jina-based**: Models from Jina (name starts with "jina")
- **Other**: Default to `SentenceTransformer`

### 2. Update the Reranker Class

Edit [retriever/Reranker.py](retriever/Reranker.py) and add your condition to the `__init__` method:

```python
def __init__(self, model_name, hf_cache_dir, use_fp16=True, normalize=True):
    # ... existing code ...
    
    if model_name in ["BAAI/bge-reranker-v2-m3"]:
        self.reranker = FlagEmbeddingReranker(model_name, cache_dir=hf_cache_dir, use_fp16=self.use_fp16)
    elif model_name.startswith("Qwen"):
        self.reranker = Qwen3Reranker(model_name, cache_dir=hf_cache_dir, use_fp16=self.use_fp16)
    elif model_name.startswith("jina"):
        self.reranker = JinaReranker(model_name, cache_dir=hf_cache_dir, use_fp16=self.use_fp16)
    elif model_name.startswith("your-new-model"):  # ADD THIS
        self.reranker = YourNewReranker(model_name, cache_dir=hf_cache_dir, use_fp16=self.use_fp16)
    else:
        self.reranker = SentenceTransformerReranker(model_name, cache_dir=hf_cache_dir, use_fp16=self.use_fp16)
```

### 3. Implement the Reranker Backend (if needed)

If your reranker doesn't fit the existing backends, create a new class in the same file:

```python
class YourNewReranker:
    def __init__(self, model_name: str, cache_dir: str = None, use_fp16: bool = True):
        """
        Initialize your custom reranker.
        
        Args:
            model_name: Model identifier (e.g., "yourname/your-reranker")
            cache_dir: Directory to cache model files
            use_fp16: Whether to use half-precision
        """
        # Load your model here
        self.model = self._load_model(model_name, cache_dir, use_fp16)
    
    def compute_scores(self, query: str, passages: list, normalize: bool) -> list:
        """
        Compute relevance scores for passages given a query.
        
        Args:
            query: The search query
            passages: List of passage texts to score
            normalize: Whether to normalize scores to [0, 1] range
            
        Returns:
            List of scores corresponding to passages (higher = more relevant)
        """
        # Your scoring logic here
        scores = [...]
        
        if normalize:
            # Normalize scores, e.g., using sigmoid
            scores = [1 / (1 + np.exp(-score)) for score in scores]
        
        return scores
    
    def _load_model(self, model_name, cache_dir, use_fp16):
        # Your model loading logic
        pass
```

## Adding a New Retrieval Model

### 1. Update Retriever Configuration

Modify your config YAML:

```yaml
retriever:
  embedding_model: "yourname/your-embedding-model"  # Your model name
  retrieval_strategy: "semantic"
  num_docs_retrieval: 10
```

### 2. How It Works

The `RAG` class loads the embedding model in `__initialize_retriever()`:

```python
embedding_model = SentenceTransformer(
    self.config.retriever.embedding_model, 
    cache_folder=self.config.general_config.hf_cache_dir
)
```

For semantic search, the query is embedded using the model, and top documents are retrieved via Elasticsearch's KNN search. No code changes needed—just update the config!

### 3. Custom Retrieval Strategy (Advanced)

If you need a custom retrieval approach beyond BM25 or vector search, modify the `__initialize_retriever()` method in [rag.py](rag.py):

```python
def __initialize_retriever(self):
    # ... existing setup ...
    
    def custom_query(search_query: str) -> Dict:
        # Your custom query logic
        return {
            "query": {
                "your_custom_query_type": {...}
            }
        }
    
    return Retriever(
        search_url=initial_search_url,
        search_func=custom_query,  # Use your custom function
        # ... rest of parameters ...
    )
```
