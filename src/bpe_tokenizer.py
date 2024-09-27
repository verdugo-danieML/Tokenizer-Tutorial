from typing import List, Dict, Tuple
from collections import defaultdict
import re
from src.utils import read_corpus, save_vocab, load_vocab
import json

class BPETokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        self.inverse_vocab = {0: "<unk>", 1: "<s>", 2: "</s>"}
        self.merges = {}

    def train(self, corpus_dir: str):
        corpus = read_corpus(corpus_dir)
        word_freqs = defaultdict(int)
        for word in corpus.split():
            word = ' '.join(word) + ' </w>'
            word_freqs[word] += 1

        for word in word_freqs:
            for char in word.split():
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
                    self.inverse_vocab[self.vocab[char]] = char

        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self._merge_vocab(best, word_freqs)
            if len(self.vocab) >= self.vocab_size:
                break

    def _get_stats(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], word_freqs: Dict[str, int]):
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        self.merges[bigram] = replacement
        self.vocab[replacement] = len(self.vocab)
        self.inverse_vocab[self.vocab[replacement]] = replacement
        
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        word_freqs.clear()
        word_freqs.update(new_word_freqs)

    def tokenize(self, text: str) -> List[str]:
        words = text.lower().split()
        tokens = []
        for word in words:
            word = ' '.join(word) + ' </w>'
            while True:
                subwords = word.split()
                if len(subwords) == 1:
                    break
                i = 0
                while i < len(subwords) - 1:
                    bigram = ' '.join(subwords[i:i+2])
                    if bigram in self.merges:
                        subwords[i] = self.merges[bigram]
                        del subwords[i+1]
                    else:
                        i += 1
                new_word = ' '.join(subwords)
                if new_word == word:
                    break
                word = new_word
            tokens.extend(word.split())
        return tokens

    def encode(self, text: str) -> List[int]:
        """Encode the input text into token IDs."""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Decode the token IDs back into text."""
        tokens = [self.inverse_vocab.get(id, "<unk>") for id in token_ids]
        return self.detokenize(tokens)

    def detokenize(self, tokens: List[str]) -> str:
        """Detokenize the input tokens."""
        text = ''.join(tokens)
        text = text.replace("<w>", "").replace("</w>", " ").strip()
        return text
    
    def save(self, vocab_file):
        data_to_save = {
            'vocab': {repr(k): v for k, v in self.vocab.items()},
            'merges': self.merges
        }
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        print(f"\nVocabulary and merges saved to {vocab_file}")

    def load(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        self.vocab = {}
        self.inverse_vocab = {}
        for k, v in loaded_data['vocab'].items():
            token = eval(k)
            id = int(v)
            self.vocab[token] = id
            self.inverse_vocab[id] = token
        
        self.merges = loaded_data['merges']

        print(f"Loaded vocabulary with {len(self.vocab)} tokens and {len(self.merges)} merges")
