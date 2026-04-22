## Overview
The **Nós RAG Evaluation Tool** provides a framework to evaluate retrieval-augmented generation (RAG) systems, with a particular focus on the retrieval and reranking stages. It integrates multiple components to process queries, retrieve relevant contexts, and generate responses using metadata-rich datasets.

## Key Features
- **Dataset and Index Management**: Create evaluation datasets and manage Elasticsearch indices.  
- **Retrieval Evaluation**: Assess retrieval and reranking modules in RAG systems.  
- **Evaluation Metrics**:  
  - **Traditional IR Metrics**: Precision, Recall, and Mean Reciprocal Rank (MRR).  
  - **LLM-as-a-Judge**: Uses the `AtlaAI/Selene-1-Mini-Llama-3.1-8B` model to compute Context Precision, Context Recall and Faithfulness/Groundness.  
- **Visualization Tools**: Edit and visualize datasets for manual inspection.  

## Project Structure
- **datasets/**: Contains datasets used for evaluation.  
  - **News/**: Press evaluation dataset.  
  - **DOG/**: DOG evaluation dataset.  
  - **Visualization_Tools/**: Tools for editing and visualizing datasets during manual revision.  
- **es_utils/**: Scripts for creating and managing Elasticsearch indices, including index configuration examples (own README).
- **experiments**:  YAML files for defining experiments.
- **ir-metrics/**: Implements traditional IR metrics for evaluation (own README).
- **llm-as-judge/**: Evaluation scripts using an LLM as a judge (own README). 
- **rag_backend/**: Implements the RAG system, including context retrieval and reranking logic. Stores experiment configurations (own README). 
- **utils/**: Utility functions for loading and processing datasets.  

Each directory includes scripts and configuration files with examples to facilitate reproducibility.

## System Flow

The following diagram illustrates the main workflow of the Nós RAG Evaluation Tool:

![System Flow](docs/Tool_flow.png)

## Usage

### Prerequisites
- Python 3.9+  
- [Elasticsearch](https://www.elastic.co/elasticsearch/) running in Docker

### Installation
```bash
sh install.sh
```

---

## Quick Start (Standard Workflow)

Follow these steps to run the standard evaluation pipeline:

### 1. Create Elasticsearch Index
```bash
cd es_utils
sh launchers/launch_indexing_dog.sh    # For DOG dataset
# or
sh launchers/launch_indexing_press.sh  # For Press dataset
```
See [es_utils README](es_utils/readme.md) for detailed indexing instructions.

---

### 2. Configure Your Experiment
Choose a configuration file in `experiments/` folder or create a new one. See [rag_backend README](rag_backend/README.md) for configuration details.

**Example**: `experiments/paper_experiments.yaml` defines:
- Embedders (BM25, Qwen3, BGE-M3, Gemma)
- Rerankers (None, Qwen, Jina, BGE)
- LLMs (e.g., Salamandra)
- Experiment combinations to run

---

### 3. Generate Test Set with Retrieved Contexts
```bash
python generate_testset.py \
  --config experiments/paper_experiments.yaml \
  --dataset datasets/DOG/Questions/questions.json \
  --run-id 1
```

This creates `results/retrieved_dataset_*.json` files with:
- Retrieved contexts for each question
- Generated responses (if LLM configured)
- Relevance scores

---

### 4. Evaluate Results

**Traditional IR metrics** (Precision, Recall, MRR):
```bash
sh launch_evaluate_ir_traditional.sh
```

**LLM-as-a-Judge** (Context Precision, Context Recall, Faithfulness):
```bash
sh launch_llm_judge.sh
```

See [ir-metrics README](ir-metrics/readme.md) and [llm-as-judge README](llm-as-judge/readme.md) for details.

---

### 5. Aggregate Results
```bash
sh launch_aggregate_metrics.sh
```

Generates a summary report combining all evaluation metrics.

---

## Advanced: Adding a New Dataset

This section explains how to add a custom dataset to the evaluation pipeline.

### Step 1: Index Your Dataset in Elasticsearch

Prepare your dataset and create an Elasticsearch index. See [es_utils README](es_utils/readme.md) for:
- Required JSON format
- Index configuration
- Indexing commands

**Quick example**:
```bash
cd es_utils
python3 es_indexing_dog.py \
  --embedding "google/embeddinggemma-300m" \
  --index "./indexes/my_dataset/index_my_dataset.json" \
  --es_config config_elastic.yaml \
  --hf_cache_dir /path/to/cache \
  --data_path /path/to/dataset.json \
  --single_file
```

---

### Step 2: Create a Dataloader

Add to `utils/dataloader_evaluation.py`:

```python
class MyDatasetDataloader(DataloaderEvaluation):
    def load_questions_with_contexts(self, file_path: str) -> List[dict]:
        """Load questions from your dataset format."""
        questions = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                questions.append({
                    "id": item['my_id_field'],
                    "question": item['my_question_field'],
                    "context": item['my_context_field'],
                    "answer": item['my_answer_field'],
                    "source_id": item['my_source_field']
                })
        return questions
```

**Key fields required**:
- `id`: Unique question identifier
- `question`: The question text
- `context`: Ground truth passage
- `answer`: Expected answer
- `source_id`: (or equivalent) Reference to the source document

---

### Step 3: Create an Elasticsearch Adapter

Add to `es_utils/index_adapters.py`:

```python
class MyDatasetAdapter(BaseDocumentAdapter):
    """Maps retrieved Elasticsearch documents to standard fields."""
    
    def get_id(self, doc) -> str:
        return doc.get('metadata', {}).get('doc_id')
    
    def get_content(self, doc) -> str:
        return doc.get('text')
    
    def get_title(self, doc) -> str:
        return doc.get('metadata', {}).get('title', '')
    
    def get_paragraph_position(self, doc) -> int:
        return doc.get('metadata', {}).get('chunk_index', -1)
    
    def get_source_id(self, doc) -> str:
        return doc.get('metadata', {}).get('source_id')
    
    def get_score(self, doc) -> float:
        return float(doc.get('score', 0.0))
```

**Methods to implement**:
- `get_id(doc)`: Extract document ID
- `get_content(doc)`: Extract passage text
- `get_title(doc)`: Extract document title
- `get_paragraph_position(doc)`: Extract chunk/paragraph number
- `get_source_id(doc)`: Extract source identifier
- `get_score(doc)`: Extract relevance score

---

### Step 4: Register in generate_testset.py

Update the dictionaries at the top of `generate_testset.py`:

```python
elasticsearch_adapters = {
    "press": PressAdapter(),
    "dog": DOGAdapter(),
    "my_dataset": MyDatasetAdapter(),  # ADD THIS
}

dataloaders = {
    "press": PressDataloader().load_questions_with_contexts,
    "dog": DOGDataloader().load_questions_with_contexts,
    "my_dataset": MyDatasetDataloader().load_questions_with_contexts,  # ADD THIS
}
```

---

### Step 5: Update Experiment Configuration

In `experiments/my_experiment.yaml`, set the dataset name:

```yaml
general_config:
  dataset_name: my_dataset  # Must match keys from Step 4
```

---

### Step 6: Generate Test Set

Run the standard workflow from Step 2-5 in the Quick Start section above. The script will automatically use your new dataloader and adapter.

```bash
python generate_testset.py \
  --config experiments/my_experiment.yaml \
  --dataset /path/to/my_questions.json \
  --run-id 1
```

---

## Detailed Component Documentation

- [rag_backend/README.md](rag_backend/README.md) - RAG system architecture, adding retrieval/reranker models
- [es_utils/readme.md](es_utils/readme.md) - Elasticsearch setup, indexing, dataset preparation
- [ir-metrics/readme.md](ir-metrics/readme.md) - Traditional IR metrics implementation
- [llm-as-judge/readme.md](llm-as-judge/readme.md) - LLM-based evaluation metrics