# IR Metrics Module

This module contains information retrieval (IR) evaluation metrics used to assess the performance of document retrieval systems in the RAG pipeline. It supports both traditional metrics (precision, recall, MRR) and context-based entity metrics.

## Table of Contents

1. [Available Metrics](#available-metrics)
2. [How Metrics Work](#how-metrics-work)
3. [Adding a New Metric](#adding-a-new-metric)
4. [Integration with Launchers](#integration-with-launchers)
5. [Output format](#output-format)

## Available Metrics

### Traditional IR Metrics

The module currently provides three standard information retrieval metrics:

1. **Precision** - Measures what fraction of retrieved documents are relevant
   - Formula: `(Relevant Retrieved) / (Total Retrieved)`
   - Range: 0 to 1

2. **Recall** - Measures what fraction of relevant documents were retrieved
   - Formula: `(Relevant Retrieved) / (Total Relevant)`
   - Range: 0 to 1

3. **Mean Reciprocal Rank (MRR)** - Measures the position of the first relevant document
   - Formula: `1 / (rank of first relevant document)`
   - Range: 0 to 1

### Entity-Based Metrics

Context entity recall metric for evaluating if retrieved contexts contain necessary entities.


## How Metrics Work

### Metric Computation Flow

```
1. Load evaluation dataset (with ground truth references)
2. Load retrieval results (with retrieved contexts)
3. For each query/question:
   - Extract reference sources from ground truth
   - Extract retrieved sources from results
   - Compute metric(s) using reference and retrieved lists
4. Aggregate results (average across all queries)
```

### Example: Precision & Recall Computation

```python
from traditional_metrics import compute_precision, compute_recall

# Ground truth references
reference_sources = ["doc1-para0", "doc2-para1"]

# Retrieved results
retrieved_sources = ["doc1-para0", "doc3-para2", "doc2-para1"]

# Compute metrics
precision = compute_precision(reference_sources, retrieved_sources)
# Result: 2/3 = 0.67 (2 out of 3 retrieved are relevant)

recall = compute_recall(reference_sources, retrieved_sources)
# Result: 2/2 = 1.0 (all relevant documents were retrieved)
```

### Evaluation Scopes

Metrics can be computed at two different levels:

1. **Paragraph-level** - Treats each paragraph as a distinct unit
   - Source ID format: `source_id-paragraph_position`
   - More granular evaluation

2. **Document-level** - Treats entire documents as units
   - Source ID format: `source_id`
   - Deduplicates paragraph results to document level

## Adding a New Metric

### Step 1: Implement the Metric Function

Create the metric computation function in a new file or add to `traditional_metrics.py`:

```python
# In ir-metrics/my_new_metric.py

def compute_my_metric(expected_ids: List[str], retrieved_ids: List[str], k: int = None) -> float:
    """Compute my custom metric.
    
    Args:
        expected_ids (List[str]): Ground truth source IDs
        retrieved_ids (List[str]): Retrieved source IDs
        k (int, optional): Limit to top-k results
    
    Returns:
        float: Metric score (typically 0-1)
    """
    if not retrieved_ids or not expected_ids:
        return 0.0
    
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    
    # Your metric logic here
    matching = sum(1 for id in retrieved_ids if id in expected_ids)
    score = matching / len(retrieved_ids)
    
    return score
```

### Step 2: Integrate into evaluate_ir_metrics.py

Modify the `evaluate_retrieval()` function to include your new metric:

```python
from my_new_metric import compute_my_metric

def evaluate_retrieval(eval_dataset, method='paragraph', logging=False):
    results = {
        'precision': [],
        'recall': [],
        'mrr': [],
        'my_new_metric': []  # Add your metric
    }
    for eval_item in eval_dataset:
        # ... existing code ...
        
        # Compute your metric
        my_metric_score = compute_my_metric(reference_sources, retrieved_sources)
        results['my_new_metric'].append(my_metric_score)
        
        # ... rest of code ...
    
    # Update average calculation
    avg_results = {
        'avg_precision': ...,
        'avg_recall': ...,
        'avg_mrr': ...,
        'avg_my_new_metric': sum(results['my_new_metric']) / len(results['my_new_metric']) 
                            if results['my_new_metric'] else 0,
    }
    return avg_results
```

### Step 3: Add Command-Line Arguments (Optional)

If your metric should be selectable via command line, modify `evaluate_ir_metrics.py`:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(...)
    # ... existing arguments ...
    parser.add_argument('--metrics', type=str, nargs='+', 
                        choices=['precision', 'recall', 'mrr', 'my_new_metric'],
                        help='Metrics to compute')
    args = parser.parse_args()
```

## Integration with Launchers

The IR metrics module is called from the main launcher script: `launch_evaluate_ir_traditional.sh`

### Current Launcher Structure

```bash
#!/bin/bash
source /home/compartido/pabloF/load_env.sh

BASE_DIR="nos-rag-eval/results/generations_retrievals"

for EXP_DIR in "$BASE_DIR"/*/; do
    echo "Procesando directorio: $EXP_DIR"
    OUTFILE="$EXP_DIR/traditional_metric_results.jsonl"
    
    cd ir-metrics/
    python3 evaluate_ir_metrics.py --folder "$EXP_DIR" --output "$OUTFILE" --scope "document"
    cd ..
done
```

### Modifying the Launcher for New Metrics

If your metric should be evaluated automatically, update the launcher:

```bash
#!/bin/bash
source /home/compartido/pabloF/load_env.sh

BASE_DIR="nos-rag-eval/results/generations_retrievals"

for EXP_DIR in "$BASE_DIR"/*/; do
    echo "Procesando directorio: $EXP_DIR"
    OUTFILE="$EXP_DIR/metric_results.jsonl"
    
    cd ir-metrics/
    # If you added --metrics argument:
    python3 evaluate_ir_metrics.py \
        --folder "$EXP_DIR" \
        --output "$OUTFILE" \
        --scope "document" \
        --metrics precision recall mrr my_new_metric
    cd ..
done
```

### Running the Launcher

```bash
bash launch_evaluate_ir_traditional.sh
```

Results are saved to `results/generations_retrievals/<exp_dir>/traditional_metric_results.jsonl`


## Output Format

The evaluation produces output in JSONL format (one JSON object per line):

```json
{
  "file": "retrieval_results.json",
  "avg_precision_paragraph": 0.75,
  "avg_recall_paragraph": 0.82,
  "avg_mrr_paragraph": 0.65,
  "avg_precision_document": 0.88,
  "avg_recall_document": 0.91,
  "avg_mrr_document": 0.78
}
```

If you add a custom metric, it will be included in this output:

```json
{
  "file": "retrieval_results.json",
  "avg_precision_paragraph": 0.75,
  "avg_my_new_metric_paragraph": 0.80,
  ...
}
```