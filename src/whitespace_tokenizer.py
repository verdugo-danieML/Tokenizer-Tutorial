from collections import defaultdict
from typing import List
from .tokenizer_base import Tokenizer

class WhitespaceTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.vocab = {}
        self.inverse_vocab = {}

    def train(self, text: str, vocab_size: int = None):
        words = text.split()
        unique_words = set(words)
        if vocab_size is not None and vocab_size < len(unique_words):
            word_counts = defaultdict(int)
            for word in words:
                word_counts[word] += 1
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            unique_words = [word for word, _ in sorted_words[:vocab_size]]
        
        self.vocab = {word: i for i, word in enumerate(unique_words)}
        self.inverse_vocab = {i: word for word, i in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(word, len(self.vocab)) for word in text.split()]

    def decode(self, ids: List[int]) -> str:
        return ' '.join(self.inverse_vocab.get(id, '[UNK]') for id in ids)