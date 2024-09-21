import re
from collections import defaultdict
from typing import List, Dict
from .tokenizer_base import Tokenizer

class RegexTokenizer(Tokenizer):
    PATTERNS = {
        'basic': r'\b\w+\b|\S',
        'gpt2': r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        'gpt4': r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    }

    def __init__(self, pattern_name='basic'):
        super().__init__()
        if pattern_name not in self.PATTERNS:
            raise ValueError(f"Unknown pattern name. Choose from: {', '.join(self.PATTERNS.keys())}")
        self.pattern = re.compile(self.PATTERNS[pattern_name], re.UNICODE)
        self.vocab = {}
        self.inverse_vocab = {}

    def train(self, text: str, vocab_size: int = None):
        tokens = self.pattern.findall(text)
        token_counts = defaultdict(int)
        for token in tokens:
            token_counts[token] += 1
        
        if vocab_size is None or vocab_size >= len(token_counts):
            unique_tokens = set(tokens)
        else:
            sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
            unique_tokens = [token for token, _ in sorted_tokens[:vocab_size]]
        
        self.vocab = {token: i for i, token in enumerate(unique_tokens)}
        self.inverse_vocab = {i: token for token, i in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(token, len(self.vocab)) for token in self.pattern.findall(text)]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.inverse_vocab.get(id, '[UNK]') for id in ids)