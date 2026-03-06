# Code from https://discuss.huggingface.co/t/use-ragas-with-huggingface-llm/75769
import argparse
import sys
import os
import json
from tqdm import tqdm

from es_utils.index_adapters import PressAdapter, DOGAdapter
from rag_backend.rag_retriever import RAG
from utils.dataloader_evaluation import load_questions_with_metadata, load_dog_questions_with_metadata
from utils.ConfigLoader import ExperimentsLoader

elasticsearch_adapters = {
    "press": PressAdapter(),
    "dog": DOGAdapter(),
}

dataloaders = {
    "press": load_questions_with_metadata,
    "dog": load_dog_questions_with_metadata,
}

parser = argparse.ArgumentParser(description="Generate test set with RAG retriever.")
parser.add_argument('--config', type=str, default=None, help='Path to RAG config file (optional)')
parser.add_argument('--run-id', type=str, default=None, help='Optional run identifier (e.g., 1, 2, ...)')
parser.add_argument('--dataset', type=str, default=None, help='Path to the questions dataset JSON file')
args = parser.parse_args()

# Pass config file if provided
if not args.config:
    exit("Please provide a config file with --config argument.")

experiments = ExperimentsLoader.load(args.config) #All experiments use the same dataset, so we can just load the dataloader for the first one. We will loop through all experiments later to generate the retrieved dataset for each of them.
data_adapter = elasticsearch_adapters.get(experiments[0].dataset_name)
dataloader_func = dataloaders.get(experiments[0].dataset_name)
dataset = []
dataset_path = args.dataset
if dataset_path and os.path.exists(dataset_path):
    print(f"Loading questions from {dataset_path}...")
    dataset = dataloader_func(file_path=dataset_path)
else:
    exit("No valid dataset path provided")

for exp_conf in experiments:
    print(f"Using config file saved in {exp_conf}...")
    rag = RAG(config=exp_conf)
    print(exp_conf)
    if args.run_id:
        output_file = f'results/retrieved_dataset_{exp_conf.name}_run{args.run_id}.json'
    else:
        output_file = f'results/retrieved_dataset_{exp_conf.name}.json'
        
    # Initialize or load existing results
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        # Get the last processed id
        processed_ids = {item['id'] for item in results}
    else:
        results = []
        processed_ids = set()

    try:
        for item in tqdm(dataset, desc="Generating dataset"):
            #print(item)
            #exit(1)
            idx = item['id']
            if idx in processed_ids:
                continue    
            query = item['question']
            relevant_docs = []
            try:
                relevant_docs = rag.retrieve_contexts(query)
            except Exception as e:
                print(f"Error retrieving contexts for query {idx}: {str(e)}")
            
            try:
                retrieved_contexts = []
                for doc in relevant_docs:
                    print(doc)
                    retrieved_contexts.append({
                        "context": data_adapter.get_content(doc),
                        "score": data_adapter.get_score(doc),
                        "context_metadata": {
                            "id": data_adapter.get_id(doc),
                            "source_id": data_adapter.get_source_id(doc),
                            "title": data_adapter.get_title(doc),
                            "paragraph_position": data_adapter.get_paragraph_position(doc),
                        }
                    })
                    #print(f"Retrieved context for query {idx}: {doc['content'][:100]}... with score {doc['score']} and metadata {metadata}")
                # Create new result
                new_result = {
                    "id": idx,
                    "user_input": query,
                    "reference_source_id": item.get('source_id') or item.get('file_name'),
                    "reference_context": item.get('context',''),
                    "reference_context_paragraphs": item.get('context_paragraph_indices',None),
                    #"answer_reference": item['answer'],
                    "retrieved_contexts": retrieved_contexts
                }
                
                # Append to results and save immediately
                results.append(new_result)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"Error during the conversion of item {idx}: {str(e)}")
                continue

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Partial results have been saved.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        print(f"\nResults saved to {output_file}")