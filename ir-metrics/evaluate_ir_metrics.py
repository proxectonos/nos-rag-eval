from traditional_metrics import compute_precision, compute_recall, compute_mrr
import json
import argparse
import os

def evaluate_retrieval(eval_dataset, method='paragraph', logging=False):
    results = {
        'precision': [],
        'recall': [],
        'mrr': []
    }
    for eval_item in eval_dataset:
        if method == 'paragraph':
            reference_sources = [f"{eval_item['reference_source_id']}-{ref_paragraph}" 
                                 for ref_paragraph in eval_item['reference_context_paragraphs']]
            retrieved_sources = [f"{ctx['context_metadata']['source_id']}-{ctx['context_metadata']['paragraph_position']}" 
                                 for ctx in eval_item['retrieved_contexts']]
            deduplicate = False
        elif method == 'document':
            reference_sources = [eval_item['reference_source_id']]
            retrieved_sources = [f"{ctx['context_metadata']['source_id']}" 
                                 for ctx in eval_item['retrieved_contexts']]
            print(f"Reference sources: {reference_sources} Retrieved sources: {retrieved_sources}")
            deduplicate = True 

        precision = compute_precision(reference_sources, retrieved_sources, deduplicate=deduplicate)
        recall = compute_recall(reference_sources, retrieved_sources, deduplicate=deduplicate)
        mrr = compute_mrr(reference_sources, retrieved_sources)
        print(f"Item ID: {eval_item['id']} Precision: {precision:.4f} Recall: {recall:.4f} MRR: {mrr:.4f}")
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['mrr'].append(mrr)
    
    # Calculate averages
    avg_results = {
        'avg_precision': sum(results['precision']) / len(results['precision']) if results['precision'] else 0,
        'avg_recall': sum(results['recall']) / len(results['recall']) if results['recall'] else 0,
        'avg_mrr': sum(results['mrr']) / len(results['mrr']) if results['mrr'] else 0
    }
    return avg_results


def evaluate_file(results_path, logging=False, scope='all'):
    with open(results_path) as f:
        eval_dataset = json.load(f)
    results = {
        "file": os.path.basename(results_path)
    }
    if scope in ['paragraph','all']:
        results_paragraph = evaluate_retrieval(eval_dataset, method='paragraph', logging=logging)
        results.update({
            "avg_precision_paragraph": results_paragraph["avg_precision"],
            "avg_recall_paragraph": results_paragraph["avg_recall"],
            "avg_mrr_paragraph": results_paragraph["avg_mrr"],
        })
    if scope in ['document','all']:
        results_document = evaluate_retrieval(eval_dataset, method='document', logging=logging)
        results.update({
            "avg_precision_document": results_document["avg_precision"],
            "avg_recall_document": results_document["avg_recall"],
            "avg_mrr_document": results_document["avg_mrr"],
        })
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate with traditional metrics Retrieval Results")
    parser.add_argument('--results', type=str, default=None, help='Path to a single results file')
    parser.add_argument('--folder', type=str, default=None, help='Path to a folder with multiple results files')
    parser.add_argument('--output', type=str, default="traditional_metric_results.jsonl", help='Output file for folder mode')
    parser.add_argument('--logging', action='store_true', help='Enable debug mode')
    parser.add_argument('--scope', type=str, choices=['paragraph', 'document', 'all'], default='all', help='Scope of evaluation: paragraph-level, document-level, or both')
    args = parser.parse_args()

    if args.folder:
        output_path = args.output
        for filename in sorted(os.listdir(args.folder)):
            if filename.endswith('.json'):
                results_path = os.path.join(args.folder, filename)
                print(f"Evaluating file: {results_path}")
                result = evaluate_file(results_path, logging=args.logging, scope=args.scope)
                with open(output_path, "a") as out_f:
                    out_f.write(json.dumps(result) + "\n")
    elif args.results:
        result = evaluate_file(args.results, logging=args.logging, scope=args.scope)
        print(json.dumps(result, indent=2))
    else:
        print("Please provide either --results <file> or --folder <folder> argument.")
