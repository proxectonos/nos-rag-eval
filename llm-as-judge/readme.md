# LLM-as-Judge Module

This module uses Large Language Models (LLMs) as judges to evaluate RAG system outputs. Instead of traditional metrics, it leverages the reasoning capabilities of LLMs to assess the quality of retrieved contexts and generated answers according to specific criteria.

## Table of Contents

1. [Available Judge Models](#available-judge-models)
2. [Available Metrics](#available-metrics)
3. [Adding a New Judge Model](#adding-a-new-judge-model)
4. [Adding a New Metric](#adding-a-new-metric)
5. [Prompt Engineering Tips](#prompt-engineering-tips)
6. [Integration with Launchers](#integration-with-launchers)
7. [Performance Considerations](#performance-considerations)

## Available Judge Models


### Selene Judge

Uses the Selene model (AtlaAI/Selene-1-Mini-Llama-3.1-8B) as the evaluation judge.

```bash
python3 judge_evaluator.py --judge_model selene ...
```

**Characteristics:**
- Smaller, more efficient model
- Good balance of speed and quality
- Default judge model
- Supports structured response parsing

### GPT Judge

Uses OpenAI's GPT-based model (gpt-oss-20b) as the evaluation judge.

```bash
python3 judge_evaluator.py --judge_model gpt ...
```

**Characteristics:**
- Larger model with advanced reasoning
- May require more resources
- Better at complex evaluation tasks

## Available Metrics

### Context Recall

Measures what fraction of ground truth information is covered by retrieved contexts.

- **How it works**: Splits reference answer into sentences, checks if each sentence is supported by at least one retrieved context
- **Formula**: `(Sentences supported by context) / (Total sentences in reference)`
- **Range**: 0 to 1
- **Use case**: Evaluate if retriever found all necessary information

### Context Precision

Measures what fraction of retrieved contexts actually contain relevant information.

- **How it works**: Computes precision@k for each retrieved context, then averages them
- **Formula**: `Mean of Precision@k for k=1 to N`
- **Range**: 0 to 1
- **Use case**: Evaluate quality of retrieved contexts, minimize noise

### Faithfulness

Measures whether the generated answer is supported by the retrieved contexts.

- **How it works**: Splits generated answer into sentences, checks if each is supported by retrieved contexts
- **Formula**: `(Answer sentences supported by context) / (Total sentences in answer)`
- **Range**: 0 to 1
- **Use case**: Evaluate answer hallucinations and factual accuracy


## Adding a New Judge Model

### Step 1: Create Judge Subclass

Add a new judge class in `Judge.py`:

```python
class MyCustomJudge(Judge):
    """Custom judge using your specific LLM model."""
    
    def __init__(self, cache_dir, device='cpu', quantization=False):
        # Initialize with your model
        model_name = "your-organization/your-model-name"
        super().__init__(model_name, cache_dir, device, quantization)
    
    def parse_response(self, response):
        """Parse model-specific response format.
        
        Args:
            response (str): Raw response from the judge model
        
        Returns:
            dict: Parsed response with 'answer' and 'confidence' keys
        """
        # Your parsing logic here
        # Example: extract yes/no answer and confidence score
        answer = "yes" in response.lower()
        return {
            "answer": answer,
            "reasoning": response
        }
```

### Step 2: Update Model Selection Logic

Modify `judge_evaluator.py` to support your new judge:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(...)
    parser.add_argument('--judge_model', type=str, 
                        choices=['gpt', 'selene', 'my_custom_judge'],  # Add here
                        default='selene', 
                        help='LLM model to use as judge')
    args = parser.parse_args()
    
    # ... in evaluation setup ...
    if args.judge_model == 'gpt':
        judge_llm = GPTJudge(cache_dir=args.cache_dir, device=device)
    elif args.judge_model == 'selene':
        judge_llm = SeleneJudge(cache_dir=args.cache_dir, device=device)
    elif args.judge_model == 'my_custom_judge':
        judge_llm = MyCustomJudge(cache_dir=args.cache_dir, device=device)
    else:
        raise ValueError("Unsupported judge model.")
```

## Adding a New Metric

### Step 1: Create Metric Function

Add metric computation function in `judge_metrics.py`:

```python
from prompts import MY_NEW_METRIC_PROMPT  # We'll create this next

def build_my_new_metric_prompt(context, reference, additional_info):
    """Build prompt for custom metric evaluation."""
    return MY_NEW_METRIC_PROMPT.format(
        context=context,
        reference=reference,
        additional_info=additional_info
    )

def compute_my_new_metric(judge, contexts, reference):
    """Compute custom metric using LLM judge.
    
    Args:
        judge: The LLM judge object
        contexts (List[str]): Retrieved context chunks
        reference (str): Reference answer or question
    
    Returns:
        float: Metric score between 0 and 1
    """
    if not contexts:
        return 0.0
    
    relevant_count = 0
    for ctx in contexts:
        prompt = build_my_new_metric_prompt(ctx, reference, "")
        result = judge.evaluate(prompt)
        
        # Parse judge response
        is_relevant = "yes" in result.lower()
        if is_relevant:
            relevant_count += 1
    
    # Aggregate to get final score
    return relevant_count / len(contexts) if contexts else 0.0
```

### Step 2: Add Prompt Template

Add to `prompts.py`:

```python
MY_NEW_METRIC_PROMPT = """
You are an expert evaluator. Given the context and reference, evaluate the metric.

Context: {context}

Reference: {reference}

Additional Info: {additional_info}

Question: Does the context satisfy the evaluation criteria?

Answer with only "yes" or "no":
"""
```

### Step 3: Register the prompt template
Add to `prompts.py`:
```python
__all__ = ['CONTEXT_RECALL_PROMPT', 'CONTEXT_PRECISION_PROMPT', 'FAITHFULNESS_PROMPT', 'MY_NEW_METRIC_PROMPT']

```

### Step 4: Integrate into Evaluator

Update `judge_evaluator.py`:

```python
from judge_metrics import (
    compute_context_recall, 
    compute_context_precision, 
    compute_faithfulness,
    compute_my_new_metric  # Add import
)

def evaluate_file(dataset, references_path, results_path, judge_llm, metric="recall"):
    """..."""
    # ... existing code ...
    
    if metric == "recall":
        score = compute_context_recall(judge_llm, retrieved_contexts, reference_response)
    elif metric == "precision":
        score = compute_context_precision(judge_llm, retrieved_contexts, user_input, reference_response)
    elif metric == "faithfulness":
        score = compute_faithfulness(judge_llm, retrieved_contexts, generated_response)
    elif metric == "my_new_metric":  # Add here
        score = compute_my_new_metric(judge_llm, retrieved_contexts, reference_response)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    return score
```

### Step 5: Add Command-Line Support

Update argument parser:

```python
parser.add_argument('--metric', type=str, 
                    choices=['recall', 'precision', 'faithfulness', 'my_new_metric'],
                    default='recall',
                    help='Metric to evaluate')
```

## Prompt Engineering Tips

Good prompts are crucial for judge accuracy. When adding new metrics:

1. **Be Explicit**: Clearly state the evaluation task
2. **Give Examples**: Provide few-shot examples if the task is complex
3. **Constrain Output**: Ask for simple yes/no answers (easier to parse)
4. **Clarify Context**: Explain what counts as relevant/irrelevant
5. **Test Variations**: Try different prompt phrasings
6. **Be Concise**: Keep prompts focused to avoid confusion

Example good prompt:

```
You are evaluating if a context paragraph contains information needed to answer a question.

Question: {question}
Context: {context}

Is the context helpful for answering the question? Answer only "yes" or "no".
```

## Integration with Launchers

The LLM-as-Judge module is called from `launch_llm_judge.sh` in the main folder.

### Current Launcher Structure

```bash
#!/bin/bash
source /home/compartido/pabloF/load_env.sh

BASE_DIR="/home/compartido/pabloF/nos-rag-eval/results/generations_retrievals"
CACHE=/home/compartido/pabloF/cache
JUDGE="selene"
DATASET="press"
REFERENCES=/home/compartido/pabloF/nos-rag-eval/datasets/News/Questions/nos-rag-dataset_questions.json

ARGS=(--dataset "$DATASET" --references "$REFERENCES" --judge_model "$JUDGE" --cache_dir "$CACHE")

for EXP_DIR in "$BASE_DIR"; do
    cd llm-as-judge/
    
    # Evaluate faithfulness
    python3 judge_evaluator.py "${ARGS[@]}" \
        --folder "$EXP_DIR" \
        --output "$EXP_DIR/judge_faithfulness.jsonl" \
        --metric faithfulness
    
    cd ..
done
```

### Modifying the Launcher for New Metrics

Add your metric to the launcher:

```bash
#!/bin/bash
source /home/compartido/pabloF/load_env.sh

BASE_DIR="/home/compartido/pabloF/nos-rag-eval/results/generations_retrievals"
CACHE=/home/compartido/pabloF/cache
JUDGE="selene"
DATASET="press"
REFERENCES=/home/compartido/pabloF/nos-rag-eval/datasets/News/Questions/nos-rag-dataset_questions.json

ARGS=(--dataset "$DATASET" --references "$REFERENCES" --judge_model "$JUDGE" --cache_dir "$CACHE")

for EXP_DIR in "$BASE_DIR"; do
    cd llm-as-judge/
    
    # Evaluate recall
    python3 judge_evaluator.py "${ARGS[@]}" \
        --folder "$EXP_DIR" \
        --output "$EXP_DIR/judge_recall.jsonl" \
        --metric recall
    
    # Evaluate precision
    python3 judge_evaluator.py "${ARGS[@]}" \
        --folder "$EXP_DIR" \
        --output "$EXP_DIR/judge_precision.jsonl" \
        --metric precision
    
    # Evaluate faithfulness
    python3 judge_evaluator.py "${ARGS[@]}" \
        --folder "$EXP_DIR" \
        --output "$EXP_DIR/judge_faithfulness.jsonl" \
        --metric faithfulness
    
    # Evaluate your new metric
    python3 judge_evaluator.py "${ARGS[@]}" \
        --folder "$EXP_DIR" \
        --output "$EXP_DIR/judge_my_new_metric.jsonl" \
        --metric my_new_metric
    
    cd ..
done
```

### Running the Launcher

```bash
bash launch_llm_judge.sh
```

Results are saved to:
- `results/generations_retrievals/<exp_dir>/judge_recall.jsonl`
- `results/generations_retrievals/<exp_dir>/judge_precision.jsonl`
- `results/generations_retrievals/<exp_dir>/judge_faithfulness.jsonl`
- `results/generations_retrievals/<exp_dir>/judge_my_new_metric.jsonl`

## Performance Considerations

### Memory Usage

- **GPT Judge**: ~20GB memory
- **Selene Judge**: ~8GB memory with quantization, ~16GB without
- Enable quantization for limited memory: `--quantize`

### Speed

- Context Recall: Slower (evaluates each sentence against each context)
- Context Precision: Medium (evaluates each context)
- Faithfulness: Slowest (evaluates each answer sentence against contexts)

### Tips for Faster Evaluation

1. Use Selene instead of GPT
2. Enable quantization: reduces memory and speeds up inference
3. Increase batch size if memory allows
4. Use GPU: significantly faster than CPU
5. Reduce max_tokens in generation
