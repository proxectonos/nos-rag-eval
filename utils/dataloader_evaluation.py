import json
from typing import List, Tuple

def load_qa_with_metadata(file_path: str):
    """
    Load questions, answers, and IDs from JSON file containing an array of QA pairs.
    Returns a list of dict (ids, questions, answers)
    """
    qa_dict = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            if isinstance(item['answer'], list):
                for i,ans in enumerate(item['answer']):
                    qa_dict.append({
                            "id": f"{item['id']}_{i}",
                            "question": item['question'],
                            "answer": ans,
                            "source_id": item['source_id'],
                            "context": item['context']
                    })
            else:
                qa_dict.append({
                        "id": f"{item['id']}_0",
                        "question": item['question'],
                        "answer": item['answer'],
                        "source_id": item['source_id'],
                        "context": item['context']
                })      
    return qa_dict

def load_questions_with_metadata(file_path: str) -> List[dict]:
    """
    Load questions and metadata from JSON file containing an array of questions.
    Returns a list of dicts with 'id', 'question', and 'metadata'.
    """
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

def load_dog_qa_with_metadata(file_path: str) -> List[dict]:
    """
    Load questions, answers, and metadata from a DOG JSON file.
    Questions and answers are always arrays — each (question, answer) pair
    is expanded into a separate entry, cross-joined by their indices.
    Returns a list of dicts with 'id', 'question', 'answer', 'file_name', 'url', 'context', 'category'.
    """
    qa_dict = []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            questions = item['question']
            answers = item['answer']
            for q_i, question in enumerate(questions):
                for a_i, answer in enumerate(answers):
                    qa_dict.append({
                        "id": f"{item['uid']}_{q_i}_{a_i}",
                        "question": question,
                        "answer": answer,
                        "file_name": item['file_name'],
                        "url": item['url'],
                        "context": item['context'],
                        "category": item['category']
                    })

    return qa_dict

def load_dog_questions_with_metadata(file_path: str) -> List[dict]:
    """
    Load questions and metadata from a DOG JSON file.
    Since each item has multiple questions, each is expanded into a separate entry.
    Returns a list of dicts with 'id', 'question', 'file_name', 'url', 'context', 'category'.
    """
    questions = []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            for q_i, question in enumerate(item['question']):
                questions.append({
                    "id": f"{item['uid']}_{q_i}",
                    "question": question,
                    "file_name": item['file_name'],
                    "url": item['url'],
                    "context": item['context'],
                    "category": item['category']
                })

    return questions
