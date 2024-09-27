import os
from typing import List, Dict
from collections import Counter

def read_corpus(directory: str) -> str:
    """Read all .txt files in the given directory and return their contents as a single string."""
    corpus = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                corpus.append(file.read())
    return " ".join(corpus)

def get_vocab(tokens: List[str]) -> Dict[str, int]:
    """Create a vocabulary from a list of tokens."""
    return dict(Counter(tokens))

def save_vocab(vocab: Dict[str, int], filename: str) -> None:
    """Save vocabulary to a file."""
    with open(filename, "w", encoding="utf-8") as f:
        for token, count in vocab.items():
            f.write(f"{token}\t{count}\n")

def load_vocab(filename: str) -> Dict[str, int]:
    """Load vocabulary from a file."""
    vocab = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                token, count = parts
                vocab[token] = int(count)
            elif len(parts) == 1:
                # If there's only one part, assume it's the token with a count of 1
                vocab[parts[0]] = 1
    return vocab