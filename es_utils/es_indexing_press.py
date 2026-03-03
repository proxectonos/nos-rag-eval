from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml

def load_elastic_config(file_path='config_elastic.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

class ElasticSearchProxy():
    def __init__(self, elastic_config_path, index_path, embedding_model, hf_cache_dir, chunking=False):
        elastic_config = load_elastic_config(elastic_config_path)
        self.es = Elasticsearch(
            hosts=[elastic_config["api_endpoint"]],
            basic_auth=(elastic_config["username"], elastic_config["password"]),
        )    
        with open(index_path, 'r', encoding='utf-8') as f:
            index_config = json.load(f)
        self.index = index_config["index_name"]
        self.mapping = index_config["mapping"]
        self.chunking = chunking
        if embedding_model:
            self.embedding_model = SentenceTransformer(embedding_model, cache_folder=hf_cache_dir)
            self.embedding_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.embedding_model = None

    def create_index(self):
        if not self.es.indices.exists(index=self.index):
            self.es.indices.create(index=self.index, body=self.mapping)
            print(f"Created index {self.index} with mapping")

    def split_text_in_chunks(self, text, chunk_size=200, overlap=50):
        """
        Splits text into chunks of chunk_size with optional overlap.
        Returns a list of text chunks.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    def index_article(self, article):
        if self.embedding_model:
            article['text_embedding'] = self.embedding_model.encode(article['text'])
        self.es.index(index=self.index, document=article)

    def index_article_with_chunks(self, article, chunk_size=200, overlap=50):
        """
        Index an article by splitting its text into chunks.
        """
        chunks = self.split_text_in_chunks(article['text'], chunk_size, overlap)
        total_chunks = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk_article = article.copy()
            chunk_article['text'] = chunk
            chunk_article['relative_chunk_id'] = idx
            chunk_article['total_chunks'] = total_chunks
            if self.embedding_model:
                chunk_article['text_embedding'] = self.embedding_model.encode(chunk)
            self.es.index(index=self.index, document=chunk_article)
    
    def index_article_with_paragraphs(self, article):
        """
        Index an article by splitting its text into paragraphs.
        """
        paragraphs = article['text'].split('\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()] # Remove empty paragraphs
        total_paragraphs = len(paragraphs)
        for idx, paragraph in enumerate(paragraphs):
            paragraph_article = article.copy()
            paragraph_article['text'] = paragraph
            paragraph_article['relative_chunk_id'] = idx
            paragraph_article['total_chunks'] = total_paragraphs
            if self.embedding_model:
                paragraph_article['text_embedding'] = self.embedding_model.encode(paragraph)
            self.es.index(index=self.index, document=paragraph_article)
        
    def index_json_files(self, data_path):
        for json_file in data_path.glob("*.jsonl"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    # Count lines for tqdm
                    total_lines = sum(1 for line in f if line.strip())
                    f.seek(0)  # Reset file pointer to beginning
                    # Create progress bar
                    pbar = tqdm(total=total_lines, desc=f"Indexing {json_file.name}")
                    # Choose the appropriate indexing function based on chunking
                    if self.chunking == 'overlap': # Overlapping chunks (200 words with 50 word overlap)
                        index_function = self.index_article_with_chunks
                    elif self.chunking == 'paragraph': # Paragraph chunking (no overlap, just paragraphs)
                        index_function = self.index_article_with_paragraphs
                    else: # No chunking
                        index_function = self.index_article
                    for line in f:
                        if line.strip():  # Skip empty lines
                            article = json.loads(line)
                            if self.chunking:
                                index_function(article)
                            else:
                                index_function(article) 
                            pbar.update(1)
                    pbar.close()     
                print(f"\nSuccessfully processed {json_file.name}")
            except json.JSONDecodeError as e:
                print(f"Error: {json_file.name} contains invalid JSON: {e}")
            except Exception as e:
                print(f"Error processing {json_file.name}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Elasticsearch index with embeddings.")
    parser.add_argument('--elastic_config', type=str, default="config/elastic_config.json", help='Path to the Elasticsearch configuration file')
    parser.add_argument('--index', type=str, default=None, help='Path to the Elasticsearch index')
    parser.add_argument('--embedding', type=str, default=None, help='Path to the embedding model')
    parser.add_argument('--hf_cache_dir', type=str, default=None, help='Path to Hugging Face cache directory')
    parser.add_argument('--data_path', type=str, default="data/combined_datasets",help="Path to the data directory containing JSON files")
    parser.add_argument('--chunking', type=str, choices=['overlap', 'paragraph', None], default=None, help='Chunking method: "overlap" or "paragraph"')
    args = parser.parse_args()
    print(args)
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data directory not found at {data_path}")
    else:
        esp = ElasticSearchProxy(args.elastic_config, args.index, args.embedding, args.hf_cache_dir, args.chunking)
        print(f"Creating index {args.index} with mapping...")
        esp.create_index()
        print(f"Processing JSON files from {data_path}")
        esp.index_json_files(data_path)
        print("\nIndexing complete!")