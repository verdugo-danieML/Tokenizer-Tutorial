import re
import random
import json
import unicodedata
from collections import defaultdict
from typing import List, Dict, Tuple, Callable
from .tokenizer_base import Tokenizer

class AdvancedBPETokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = {}
        self.special_tokens = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.pre_tokenizers = []
        self.post_processors = []

    def add_pre_tokenizer(self, func: Callable[[str], List[str]]):
        self.pre_tokenizers.append(func)

    def add_post_processor(self, func: Callable[[List[int]], List[int]]):
        self.post_processors.append(func)

    def pre_tokenize(self, text: str) -> List[str]:
        for pre_tokenizer in self.pre_tokenizers:
            text = ' '.join(pre_tokenizer(text))
        return re.findall(r'\S+|\s+', text)

    def post_process(self, ids: List[int]) -> List[int]:
        for post_processor in self.post_processors:
            ids = post_processor(ids)
        return ids

    def train(self, text: str, vocab_size: int, min_frequency: int = 2):
        words = self.pre_tokenize(text)
        chars = set(''.join(words))
        self.vocab = {c: i + len(self.special_tokens) for i, c in enumerate(chars)}
        self.vocab.update(self.special_tokens)
        self.inverse_vocab = {i: c for c, i in self.vocab.items()}

        word_freqs = defaultdict(int)
        for word in words:
            word_freqs[' '.join(word) + ' </w>'] += 1

        while len(self.vocab) < vocab_size:
            pairs = self.get_stats(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_frequency:
                break

            new_token = ''.join(best_pair)
            self.vocab[new_token] = len(self.vocab)
            self.inverse_vocab[len(self.inverse_vocab)] = new_token
            self.merges[best_pair] = new_token

            word_freqs = self.merge_word_freqs(word_freqs, best_pair, new_token)

    def get_stats(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_word_freqs(self, word_freqs: Dict[str, int], pair: Tuple[str, str], new_token: str) -> Dict[str, int]:
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = new_token
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    def encode(self, text: str) -> List[int]:
        words = self.pre_tokenize(text)
        encoded = [self.special_tokens['[CLS]']]
        for word in words:
            word = ' '.join(word) + ' </w>'
            while True:
                pairs = set(zip(word.split()[:-1], word.split()[1:]))
                if not pairs & set(self.merges.keys()):
                    break
                for pair, merge in self.merges.items():
                    word = word.replace(' '.join(pair), merge)
            encoded.extend([self.vocab.get(token, self.special_tokens['[UNK]']) for token in word.split()])
        encoded.append(self.special_tokens['[SEP]'])
        return self.post_process(encoded)

    def decode(self, ids: List[int]) -> str:
        tokens = [self.inverse_vocab.get(id, '[UNK]') for id in ids if id not in [self.special_tokens['[CLS]'], self.special_tokens['[SEP]']]]
        text = ''.join(tokens).replace('</w>', ' ').strip()
        return text

    def encode_with_dropout(self, text: str, dropout_prob: float = 0.1) -> List[int]:
        words = self.pre_tokenize(text)
        encoded = [self.special_tokens['[CLS]']]
        for word in words:
            word = ' '.join(word) + ' </w>'
            while True:
                pairs = set(zip(word.split()[:-1], word.split()[1:]))
                if not pairs & set(self.merges.keys()):
                    break
                for pair, merge in self.merges.items():
                    if random.random() > dropout_prob:
                        word = word.replace(' '.join(pair), merge)
            encoded.extend([self.vocab.get(token, self.special_tokens['[UNK]']) for token in word.split()])
        encoded.append(self.special_tokens['[SEP]'])
        return self.post_process(encoded)

    def save(self, path: str):
        data = {
            'vocab': self.vocab,
            'inverse_vocab': self.inverse_vocab,
            'merges': {' '.join(k): v for k, v in self.merges.items()},
            'special_tokens': self.special_tokens
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.inverse_vocab = {int(k): v for k, v in data['inverse_vocab'].items()}
        self.merges = {tuple(k.split()): v for k, v in data['merges'].items()}
        self.special_tokens = data['special_tokens']

# Example pre-tokenization and post-processing functions
def lowercase_and_normalize(text: str) -> List[str]:
    text = unicodedata.normalize('NFKC', text.lower())
    return re.findall(r'\b\w+\b|[^\w\s]', text)

def truncate_sequence(ids: List[int], max_length: int = 512) -> List[int]:
    return ids[:max_length]