from collections import defaultdict
from typing import List, Dict, Tuple
from .tokenizer_base import Tokenizer

class BPETokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = {}

    def train(self, text: str, vocab_size: int, min_frequency: int = 2):
        words = text.split()
        chars = set(''.join(words))
        self.vocab = {c: i for i, c in enumerate(chars)}
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
        words = text.split()
        encoded = []
        for word in words:
            word = ' '.join(word) + ' </w>'
            while True:
                pairs = set(zip(word.split()[:-1], word.split()[1:]))
                if not pairs & set(self.merges.keys()):
                    break
                for pair, merge in self.merges.items():
                    word = word.replace(' '.join(pair), merge)
            encoded.extend([self.vocab.get(token, len(self.vocab)) for token in word.split()])
        return encoded

    def decode(self, ids: List[int]) -> str:
        tokens = [self.inverse_vocab.get(id, '[UNK]') for id in ids]
        return ''.join(tokens).replace('</w>', ' ').strip()