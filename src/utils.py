import os
from typing import List

def generate_corpus(file_paths: List[str]) -> str:
    corpus = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                corpus.append(file.read())
        else:
            print(f"Warning: File not found - {file_path}")
    return "\n\n".join(corpus)
