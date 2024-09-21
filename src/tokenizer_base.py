from abc import ABC, abstractmethod
import json
from typing import List

class Tokenizer(ABC):
    @abstractmethod
    def train(self, text: str, vocab_size: int):
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        pass

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'inverse_vocab': self.inverse_vocab
            }, f)

    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.inverse_vocab = {int(k): v for k, v in data['inverse_vocab'].items()}