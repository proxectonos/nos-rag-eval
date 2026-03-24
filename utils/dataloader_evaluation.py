import json
from typing import List, Tuple
from abc import ABC

class DataloaderEvaluation(ABC):
    def load_questions_with_contexts(self, file_path: str) -> List[dict]:
        pass
    
    def load_answers(self, file_path: str) -> List[dict]:
        pass

class PressDataloader(DataloaderEvaluation):

    def load_questions_with_contexts(self, file_path: str) -> List[dict]:
        questions = []
    
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                questions.append({
                    "id": item['id'],
                    "source_id": item['source_id'],
                    "question": item['question'],
                    "context": item['context'],
                    "context_paragraph_indices": item["context_paragraph_indices"]
                })
                
        return questions

    def load_answers(self, file_path: str) -> List[dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            if isinstance(item['answer'], list):
                #for i, ans in enumerate(item['answer']):
                    yield {
                        "id": f"{item['id']}_0",
                        "answer": item['answer'][0],
                        "source_id": item['source_id'],
                        "context": item['context']
                    }
            else:
                yield {
                    "id": f"{item['id']}_0",
                    "answer": item['answer'],
                    "source_id": item['source_id'],
                    "context": item['context']
                }   

class DOGDataloader(DataloaderEvaluation):
    
    def load_questions_with_contexts(self, file_path: str) -> List[dict]:
        questions = []
    
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                for q_i, question in enumerate(item['question']):
                    questions.append({
                        "id": f"{item['uid']}_{q_i}",
                        "file_name": item['file_name'],
                        "url": item['url'],
                        "category": item['category'],
                        "question": question,
                        "context": item['context']
                    })
                
        return questions

    def load_answers(self, file_path: str) -> List[dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                questions = item['question']
                answers = item['answer']
                for q_i, question in enumerate(questions):
                    yield {
                        "id": f"{item['uid']}_{q_i}",
                        "answer": answers,
                        "file_name": item['file_name'],
                        "url": item['url'],
                        "category": item['category'],
                        "context": item['context']
                    }