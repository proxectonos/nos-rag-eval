# Elasticsearch RAG Utils Documentation

## Table of Contents

1. [Create ES Container](#create-es-container)

2. [Index Existing Datasets](#index-existing-datasets)

3. [Adding a New Dataset](#adding-a-new-dataset)

4. [Index Adapters scripts](#index-adapters)
---

# Create ES Container

## Step 1: Download Elastic docker image (if it is installed in the system, move to step 2)

```bash
docker network create elastic
docker pull docker.elastic.co/elasticsearch/elasticsearch:9.2.3
```

## Step 2: Start container
```bash
docker run --name elastic-rag-eval \
  --net elastic \
  -p 9202:9200 \
  -d \
  -m 6GB \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=true" \
  -e "ES_JAVA_OPTS=-Xms3g -Xmx3g" \
  -v es_data_eval:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:9.2.3
```
Careful: the first port in the `-p` argument must be different for the port uses by another instances (see `docker ps -a` to check current instances). Be careful also with the data volume (`-v`argument). If there is any problem, change the name to de data variable (`es_data_eval` in this example).

## Step 3: Set password
```bash
docker exec -it elastic-rag-eval /usr/share/elasticsearch/bin/elasticsearch-reset-password -u elastic
```
Add this generated password in the `config_elastic.yaml` file

---

# Index Existing Datasets

This section explains how to index pre-configured datasets (Press and DOG) into Elasticsearch.

## Prerequisites

Before indexing any dataset, ensure:
1. Elasticsearch container is running (see "Create ES container" section above)
2. The `config_elastic.yaml` file contains the correct credentials and API endpoint
3. The dataset files are available in their expected locations
4. Required embedding models are available (or cached in your HF cache directory)

## Indexing Process Overview

The indexing process involves:
1. Loading dataset documents from JSON files
2. Optionally generating embeddings using a specified model
3. Splitting documents into chunks (if chunking is enabled)
4. Creating an Elasticsearch index with the appropriate mapping
5. Indexing all documents with their metadata and embeddings

## Available Datasets

### DOG Dataset

**Location**: `/home/compartido/pabloF/data/DOG/final_data/dog_2025_formated.json`

**Description**: Dictionary-like dataset with article structures

**Available Indexes**:
- `dog_2025_bm25.json` - BM25 full-text search (no embeddings)
- `dog_2025_gemma.json` - Gemma 300M embeddings
- `dog_2025_qwen3.json` - Qwen3 0.6B embeddings

**Indexing Command**:
```bash
cd es_utils
python3 es_indexing_dog.py \
    --embedding "google/embeddinggemma-300m" \
    --index "./indexes/dog/dog_2025_gemma.json" \
    --es_config config_elastic.yaml \
    --hf_cache_dir /home/compartido/pabloF/cache \
    --data_path /home/compartido/pabloF/data/DOG/final_data/dog_2025_formated.json \
    --single_file \
    --chunking "overlap" \
    --transform_dog
```

**Parameters**:
- `--embedding`: Model to use for generating embeddings (optional, for BM25 only omit this)
- `--index`: Path to the index configuration JSON file
- `--es_config`: Path to Elasticsearch configuration file
- `--hf_cache_dir`: Hugging Face models cache directory
- `--data_path`: Path to the dataset JSON file
- `--single_file`: Use this flag when data_path points to a single JSON file
- `--chunking`: Chunking method - "overlap" or "paragraph"
- `--transform_dog`: Transform DOG JSON structure (document.content) to expected format

**Quick Launch**:
```bash
cd es_utils/launchers
bash launch_indexing_dog.sh
```

### Press Dataset

**Location**: `../datasets/News/Documents/`

**Description**: Press/news articles dataset with paragraph-based structure

**Available Indexes**:
- `index_all-minilm-l6-v2_paragraph.json` - All-MiniLM-L6-v2 embeddings
- `index_bge-m3_paragraph.json` - BGE-M3 embeddings
- `index_gemma-300m_paragraph.json` - Gemma 300M embeddings
- `index_granite-english-r2_paragraph.json` - IBM Granite English R2 embeddings
- `index_qwen3_paragraph.json` - Qwen3 0.6B embeddings
- `index_no_embedding_paragraph.json` - BM25 without embeddings

**Indexing Command**:
```bash
cd es_utils
python3 es_indexing_press.py \
    --embedding "BAAI/bge-m3" \
    --index "indexes/press/index_bge-m3_paragraph.json" \
    --elastic_config config_elastic.yaml \
    --hf_cache_dir /home/compartido/pabloF/cache \
    --data_path ../datasets/News/Documents \
    --chunking "paragraph"
```

**Parameters**:
- `--embedding`: Model to use for generating embeddings (optional)
- `--index`: Path to the index configuration JSON file
- `--elastic_config`: Path to Elasticsearch configuration file
- `--hf_cache_dir`: Hugging Face models cache directory
- `--data_path`: Path to directory containing press documents
- `--chunking`: Chunking method - "paragraph" for press data

**Quick Launch**:
```bash
cd es_utils/launchers
bash launch_indexing_press.sh
```

## Monitoring Indexing Progress

During indexing, you will see:
- Progress bar showing documents processed
- Number of successfully indexed documents
- Any errors encountered during processing
- Index refresh confirmation

At the end, a summary is printed:
```
Indexing complete!
Successfully indexed: X articles
Failed to index: Y articles
```

# Adding a New Dataset

This section explains how to prepare and index a new dataset into Elasticsearch.

## Step 1: Prepare Your Dataset

### Required Data Format

Your dataset must be organized as JSON files with the following structure:

```json
{
  "articles": [
    {
      "id": "unique_identifier",
      "title": "Article Title",
      "content": "Full article text",
      "source": "Data source name",
      "type": "document_type",
      "metadata_field_1": "value1",
      "metadata_field_2": "value2"
    }
  ]
}
```

Or as individual JSON files in a directory, each containing:

```json
{
  "id": "unique_identifier",
  "title": "Article Title",
  "content": "Full article text",
  "source": "Data source name",
  "type": "document_type",
  "metadata_field_1": "value1"
}
```

### Dataset Organization

Two options:

1. **Single JSON file with articles array**: 
   - Place the file in your data directory
   - Use `--single_file` flag when indexing
   - Specify `--articles_key` if your array key is not "articles"

2. **Multiple JSON files in a directory**:
   - Place all JSON files in a directory
   - Each file represents one or more documents
   - The script will recursively process all JSON files

## Step 2: Create Index Configuration

Create a JSON configuration file for your Elasticsearch index in `indexes/your_dataset/` directory:

```json
{
    "index_name": "my_dataset_index",
    "mapping": {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "type": {"type": "keyword"},
                "source": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "content": {"type": "text", "analyzer": "standard"},
                "text_embedding": {"type": "dense_vector", "dims": 768},
                "num_words": {"type": "integer"},
                "lang": {"type": "keyword"}
            }
        }
    }
}
```

**Key considerations**:
- `index_name`: Unique name for your Elasticsearch index
- `text_embedding` field should match your embedding model's dimensions:
  - All-MiniLM-L6-v2: 384 dimensions
  - BGE-M3: 1024 dimensions
  - Gemma 300M: 768 dimensions
  - Qwen3 0.6B: 1024 dimensions
- Include only `text_embedding` field if using embeddings
- For BM25-only indexing, omit the `text_embedding` field

### Index Configuration Template

Copy from existing examples in `indexes/examples/`:
- `index_embedding_example.json` - For indexed-based search with embeddings
- `index_bm25_example.json` - For BM25 full-text search

## Step 3: Update Configuration Files

### Update config_elastic.yaml (if needed)

The `config_elastic.yaml` must contain your Elasticsearch connection details:

```yaml
username: elastic
password: <your_elasticsearch_password>
elastic_url: http://localhost:5601
api_endpoint: http://localhost:9202  # Adjust port if different
```

### Create Launch Script (optional but recommended)

Create a launch script in `launchers/launch_indexing_my_dataset.sh`:

```bash
#!/bin/bash

HF_CACHE="/home/compartido/pabloF/cache"
MY_DATASET_PATH="/path/to/your/dataset.json"
ES_CONFIG_FILE="config_elastic.yaml"

cd ..

python3 es_indexing_dog.py \
    --embedding "google/embeddinggemma-300m" \
    --index "./indexes/my_dataset/index_my_dataset.json" \
    --es_config $ES_CONFIG_FILE \
    --hf_cache_dir $HF_CACHE \
    --data_path $MY_DATASET_PATH \
    --single_file \
    --chunking "overlap"
```

Make it executable:
```bash
chmod +x launchers/launch_indexing_my_dataset.sh
```

## Step 4: Run Indexing

Use the appropriate indexing script:

### For datasets similar to DOG (single JSON file with articles array):

```bash
cd es_utils
python3 es_indexing_dog.py \
    --embedding "google/embeddinggemma-300m" \
    --index "./indexes/my_dataset/my_index.json" \
    --es_config config_elastic.yaml \
    --hf_cache_dir /home/compartido/pabloF/cache \
    --data_path /path/to/my_data.json \
    --single_file \
    --chunking "overlap"
```

### For datasets with multiple documents in a directory:

```bash
cd es_utils
python3 es_indexing_press.py \
    --embedding "BAAI/bge-m3" \
    --index "indexes/my_dataset/my_index.json" \
    --elastic_config config_elastic.yaml \
    --hf_cache_dir /home/compartido/pabloF/cache \
    --data_path /path/to/documents/directory \
    --chunking "paragraph"
```

### Without embeddings (BM25 only):

```bash
cd es_utils
python3 es_indexing_dog.py \
    --index "./indexes/my_dataset/my_index_bm25.json" \
    --es_config config_elastic.yaml \
    --hf_cache_dir /home/compartido/pabloF/cache \
    --data_path /path/to/my_data.json \
    --single_file
```

## Step 5: Verify Indexing

After indexing completes successfully:

1. Check in Kibana (if available):
   - Navigate to `http://localhost:5601`
   - Go to Stack Management → Indices
   - Verify your index appears in the list

2. Check index statistics via curl:
   ```bash
   curl -u elastic:password http://localhost:9202/my_dataset_index/_stats
   ```

3. Query the index:
   ```bash
   curl -u elastic:password http://localhost:9202/my_dataset_index/_search
   ```
---

# Index Adapters

The `index_adapters.py` file contains adapter classes that handle dataset-specific document structure variations in the RAG system. Adapters are required to retrieve information from documents in a standardized way, regardless of their internal JSON structure.

## What are Index Adapters?

Index adapters are classes that implement the `BaseDocumentAdapter` abstract interface. They define how to extract specific fields from your dataset's documents. This is essential because:

1. Different datasets may have different JSON structures
2. The RAG system needs consistent access to key fields (id, content, title, etc.)
3. Adapters abstract away the document structure complexity

## BaseDocumentAdapter Interface

All adapters must inherit from `BaseDocumentAdapter` and implement these methods:

```python
class BaseDocumentAdapter(ABC):
    @abstractmethod
    def get_id(self, doc) -> str: 
        """Extract unique document identifier"""
        pass
    
    @abstractmethod
    def get_content(self, doc) -> str: 
        """Extract document content/text"""
        pass
    
    @abstractmethod
    def get_title(self, doc) -> str: 
        """Extract document title or headline"""
        pass
    
    @abstractmethod
    def get_paragraph_position(self, doc) -> int: 
        """Extract chunk/paragraph position (for chunked documents)"""
        pass

    @abstractmethod
    def get_source_id(self, doc) -> str: 
        """Extract source identifier or document reference"""
        pass

    @abstractmethod
    def get_score(self, doc) -> float: 
        """Extract relevance score from search results"""
        pass
```

## Available Adapters

### PressAdapter

Used for press/news articles dataset with metadata structure:

```python
class PressAdapter(BaseDocumentAdapter):
    def get_id(self, doc) -> str:
        metadata = doc.get('metadata', {})
        return metadata.get('id')
    
    def get_content(self, doc) -> str:
        return doc.get('content', '')

    def get_title(self, doc) -> str:
        metadata = doc.get('metadata', {})
        return metadata.get('title') or metadata.get('headline', '')

    def get_paragraph_position(self, doc) -> int:
        metadata = doc.get('metadata', {})
        return metadata.get('relative_chunk_id', -1)

    def get_source_id(self, doc) -> str:
        metadata = doc.get('metadata', {})
        return metadata.get('source_id') or f"Praza-{metadata.get('published_on')}"
    
    def get_score(self, doc) -> float:
        metadata = doc.get('metadata', {})
        return metadata.get('score', 0.0)
```

**Expected Document Structure**:
```json
{
  "content": "Article text...",
  "metadata": {
    "id": "article_id",
    "title": "Article Title",
    "headline": "Alternative title",
    "source_id": "source_identifier",
    "published_on": "2025-04-21 10:00:00",
    "relative_chunk_id": 0,
    "score": 0.95
  }
}
```

### DOGAdapter

Used for DOG dataset with nested metadata structure:

```python
class DOGAdapter(BaseDocumentAdapter):
    def get_id(self, doc) -> str:
        meta = doc.get('metadata', {}).get('metadata', {})
        return meta.get('doga_id') or f"DOG-{meta.get('doga_date')}"

    def get_content(self, doc) -> str:
        return doc.get('content')

    def get_title(self, doc) -> str:
        document = doc.get('metadata', {}).get('document', {})
        return document.get('title', '')

    def get_paragraph_position(self, doc: dict) -> int:
        return doc.get('metadata', {}).get('relative_chunk_id', -1)

    def get_source_id(self, doc) -> str:
        meta = doc.get('metadata', {}).get('metadata', {})
        return meta.get('file_id', '')

    def get_score(self, doc) -> float:
        score = doc.get('score')
        return float(score) if score is not None else 0.0
```

**Expected Document Structure**:
```json
{
  "content": "Document content...",
  "metadata": {
    "metadata": {
      "doga_id": "unique_id",
      "doga_date": "2025-04-21",
      "file_id": "source_identifier"
    },
    "document": {
      "title": "Document Title"
    },
    "relative_chunk_id": 0
  },
  "score": 0.92
}
```

## Creating a New Adapter for Your Dataset

### Step 1: Understand Your Document Structure

First, examine your dataset documents to understand their JSON structure. For example:

```json
{
  "doc_id": "my_doc_001",
  "doc_title": "Title",
  "doc_content": "Full text...",
  "metadata": {
    "source": "my_source",
    "chunk_index": 0
  }
}
```

### Step 2: Create the Adapter Class

Add a new adapter class to `index_adapters.py`:

```python
class MyDatasetAdapter(BaseDocumentAdapter):
    
    def get_id(self, doc) -> str:
        # Extract ID from your document structure
        return doc.get('doc_id', '')
    
    def get_content(self, doc) -> str:
        # Extract content/text from your document
        return doc.get('doc_content', '')
    
    def get_title(self, doc) -> str:
        # Extract title from your document
        return doc.get('doc_title', '')
    
    def get_paragraph_position(self, doc) -> int:
        # Extract chunk position if applicable
        metadata = doc.get('metadata', {})
        return metadata.get('chunk_index', -1)
    
    def get_source_id(self, doc) -> str:
        # Extract source identifier
        metadata = doc.get('metadata', {})
        return metadata.get('source', '')
    
    def get_score(self, doc) -> float:
        # Extract score if present in document
        score = doc.get('score')
        return float(score) if score is not None else 0.0
```

### Step 3: Register the Adapter

Once created, ensure the adapter is imported where it's used in the RAG system. Typically, you'll need to add it to the imports in the main RAG backend code.

### Step 4: Use the Adapter in Your RAG System

In your RAG retrieval code, instantiate and use the adapter:

```python
from index_adapters import MyDatasetAdapter

adapter = MyDatasetAdapter()

# When processing search results
for doc in search_results:
    doc_id = adapter.get_id(doc)
    content = adapter.get_content(doc)
    title = adapter.get_title(doc)
    position = adapter.get_paragraph_position(doc)
    source = adapter.get_source_id(doc)
    score = adapter.get_score(doc)
```

## Important Considerations When Creating Adapters

1. **Field Existence**: Always use `.get()` with defaults to handle missing fields
   ```python
   return doc.get('field_name', 'default_value')
   ```

2. **Nested Structures**: Handle deeply nested dictionaries carefully
   ```python
   return doc.get('level1', {}).get('level2', {}).get('field', 'default')
   ```

3. **Type Conversion**: Ensure correct types are returned (str, int, float)
   ```python
   score = doc.get('score')
   return float(score) if score is not None else 0.0
   ```

4. **Fallback Values**: Provide sensible fallbacks for missing data
   ```python
   return metadata.get('id') or f"Unknown-{metadata.get('timestamp')}"
   ```

5. **Error Handling**: Handle unexpected document formats gracefully
   ```python
   try:
       return int(doc.get('position', -1))
   except (ValueError, TypeError):
       return -1
   ```

## Testing Your Adapter

Before deploying, test your adapter with sample documents from your dataset:

```python
from index_adapters import MyDatasetAdapter
import json

adapter = MyDatasetAdapter()

# Load a sample document
with open('sample_document.json') as f:
    sample_doc = json.load(f)

# Test each method
print(f"ID: {adapter.get_id(sample_doc)}")
print(f"Title: {adapter.get_title(sample_doc)}")
print(f"Content length: {len(adapter.get_content(sample_doc))}")
print(f"Position: {adapter.get_paragraph_position(sample_doc)}")
print(f"Source: {adapter.get_source_id(sample_doc)}")
print(f"Score: {adapter.get_score(sample_doc)}")
```

Ensure all methods return the expected values and types for your dataset.
