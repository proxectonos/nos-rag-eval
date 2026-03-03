from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
import yaml

def load_yaml_config(file_path='config_elastic.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

class ElasticSearchProxy():
    def __init__(self, es_config, index_path, embedding_model, hf_cache_dir, chunking=False):
        es_config_data = load_yaml_config(es_config)
        self.es = Elasticsearch(
            hosts=[es_config_data.get('api_endpoint')], #https://localhost:9200 if using SSL certificates
            basic_auth=(es_config_data.get('username'), es_config_data.get('password')),
            #ca_certs=os.environ["ES_HOME"] + "/config/certs/http_ca.crt", #if using SSL certificates
            #verify_certs=True, #if using SSL certificates
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
        else:
            print(f"Index {self.index} already created. Going to indexing new documents...")
            time.sleep(3)            

    def split_text_in_chunks(self, text, chunk_size=250, overlap=50):
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

    def index_article_with_chunks(self, article, batch_size=32, chunk_size=250, overlap=50):
        chunks = self.split_text_in_chunks(article["news"]["body"], chunk_size, overlap)
        total_chunks = len(chunks)
        docs = []
        
        for idx, chunk in enumerate(chunks):
            doc = article.copy()
            doc['text'] = chunk
            doc['relative_chunk_id'] = idx
            doc['total_chunks'] = total_chunks
            docs.append(doc)

        # batch embeddings
        if self.embedding_model:
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i+batch_size]
                texts = [d['text'] for d in batch]
                embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
                for j, d in enumerate(batch):
                    d['text_embedding'] = embeddings[j]

        actions = [{"_index": self.index, "_source": d} for d in docs]
        helpers.bulk(self.es, actions)
    
    def index_article_with_paragraphs(self, article, batch_size=32):
        """
        Index an article by splitting its text into paragraphs.
        """
        paragraphs = article["news"]["body"].split('\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()] # Remove empty paragraphs
        total_paragraphs = len(paragraphs)

        docs = []
        for idx, paragraph in enumerate(paragraphs):
            doc = article.copy()
            doc['text'] = paragraph
            doc['relative_chunk_id'] = idx
            doc['total_chunks'] = total_paragraphs
            docs.append(doc)    

        if self.embedding_model:
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i+batch_size]
                texts = [d['text'] for d in batch]
                embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
                for j, d in enumerate(batch):
                    d['text_embedding'] = embeddings[j]

        # Prepare actions for bulk indexing
        actions = [
            {"_index": self.index, "_source": d}
            for d in docs
        ]

        helpers.bulk(self.es, actions)

    def index_json_files(self, data_path, max_workers=8):
        files = list(data_path.rglob("*.json"))
        print(f"Found {len(files)} JSON files")

        def process_file(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                if self.chunking == 'overlap':
                    self.index_article_with_chunks(article)
                elif self.chunking == 'paragraph':
                    self.index_article_with_paragraphs(article)
                else:
                    self.index_article(article)
                return True
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
                return False

        # Executor inside the method
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(process_file, files), total=len(files)))

        # Refresh once at the end
        self.es.indices.refresh(index=self.index)
        print(f"Indexing complete. {sum(results)} files successfully indexed.")

    def transform_dog_article(self, article):
        """
        Transform DOGA JSON structure to the expected format.
        Converts structure with 'document.content' to format with 'text' and 'news.body'
        """
        transformed = article.copy()
        
        # Set 'text' field from document.content
        if 'document' in article and 'content' in article['document']:
            transformed['text'] = article['document']['content']
            
            # Create 'news' structure with 'body' for chunking functions
            if 'news' not in transformed:
                transformed['news'] = {}
            transformed['news']['body'] = article['document']['content']
        
        return transformed

    def index_single_json_file(self, json_file_path, articles_key='articles', batch_size=100, transform_dog=False):
        """
        Index articles from a single JSON file containing multiple articles.
        
        Args:
            json_file_path: Path to the JSON file
            articles_key: Key in the JSON that contains the list of articles (default: 'articles')
            batch_size: Number of articles to process at once
            transform_dog: If True, transforms DOGA structure (document.content) to expected format
        """
        print(f"Loading articles from single JSON file: {json_file_path}")
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract articles list
            if isinstance(data, list):
                articles = data
            elif isinstance(data, dict) and articles_key in data:
                articles = data[articles_key]
            else:
                raise ValueError(f"Could not find articles. Expected a list or dict with '{articles_key}' key")
            
            print(f"Found {len(articles)} articles in the JSON file")
            
            # Process articles in batches
            success_count = 0
            error_count = 0
            
            for i in tqdm(range(0, len(articles), batch_size), desc="Processing batches"):
                batch = articles[i:i+batch_size]
                
                for article in batch:
                    try:
                        # Transform if needed
                        if transform_dog:
                            article = self.transform_dog_article(article)
                        
                        if self.chunking == 'overlap':
                            self.index_article_with_chunks(article)
                        elif self.chunking == 'paragraph':
                            self.index_article_with_paragraphs(article)
                        else:
                            self.index_article(article)
                        success_count += 1
                    except Exception as e:
                        print(f"\nError processing article: {e}")
                        error_count += 1
            
            # Refresh index
            self.es.indices.refresh(index=self.index)
            print(f"\nIndexing complete!")
            print(f"Successfully indexed: {success_count} articles")
            if error_count > 0:
                print(f"Failed to index: {error_count} articles")
                
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Elasticsearch index with embeddings.")
    parser.add_argument('--es_config', type=str, default=None, help='Path to the Elasticsearch index configuration JSON file')
    parser.add_argument('--index', type=str, default=None, help='Path to the Elasticsearch index')
    parser.add_argument('--embedding', type=str, default=None, help='Path to the embedding model')
    parser.add_argument('--hf_cache_dir', type=str, default=None, help='Path to Hugging Face cache directory')
    parser.add_argument('--data_path', type=str, default="data/combined_datasets",help="Path to the data directory containing JSON files")
    parser.add_argument('--chunking', type=str, choices=['overlap', 'paragraph', None], default=None, help='Chunking method: "overlap" or "paragraph"')
    parser.add_argument('--single_file', action='store_true', help='Use this flag if data_path points to a single JSON file with multiple articles')
    parser.add_argument('--articles_key', type=str, default='articles', help='Key name for the articles list in single JSON file (default: "articles")')
    parser.add_argument('--transform_dog', action='store_true', help='Transform DOGA JSON structure (document.content) to expected format')
    args = parser.parse_args()
    print(args)
    
    if not args.es_config or not args.index:
        print("Error: --es_config and --index arguments are required.")
        exit(1)
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data path not found at {data_path}")
        exit(1)

    esp = ElasticSearchProxy(args.es_config, args.index, args.embedding, args.hf_cache_dir, args.chunking)
    print(f"Creating index {args.index} with mapping...")
    esp.create_index()      
    if args.single_file:
        print(f"Processing single JSON file: {data_path}")
        esp.index_single_json_file(data_path, articles_key=args.articles_key, transform_dog=args.transform_dog)
    else:
        print(f"Processing multiple JSON files from directory: {data_path}")
        esp.index_json_files(data_path)  
    print("\nIndexing complete!")