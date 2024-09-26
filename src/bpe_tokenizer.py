from typing import List, Dict, Tuple
from collections import defaultdict
import re
from src.utils import read_corpus, save_vocab, load_vocab

class BPETokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {"<unk>": 0, "<w>": 1, "</w>": 2}
        self.inverse_vocab = {0: "<unk>", 1: "<w>", 2: "</w>"}
        self.merges = {}

    def train(self, corpus_dir: str):
        """Train the BPE tokenizer on the given corpus."""
        corpus = read_corpus(corpus_dir)
        print(f"Corpus size: {len(corpus)} characters")
        
        word_freqs = defaultdict(int)
        for word in corpus.split():
            word = f"<w>{word}</w>"
            word_freqs[word] += 1
            for char in word:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
                    self.inverse_vocab[self.vocab[char]] = char

        while len(self.vocab) < self.vocab_size:
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self._merge_vocab(best, word_freqs)

    def _get_stats(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], word_freqs: Dict[str, int]):
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        replacement = ''.join(pair)
        self.merges[pair] = replacement
        self.vocab[replacement] = len(self.vocab)
        self.inverse_vocab[self.vocab[replacement]] = replacement
        for word in list(word_freqs.keys()):
            new_word = pattern.sub(replacement, word)
            if new_word != word:
                word_freqs[new_word] = word_freqs.pop(word)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text using the trained BPE model."""
        words = text.split()
        tokens = []
        for word in words:
            word = f"<w>{word}</w>"
            subwords = list(word)
            while True:
                if len(subwords) == 1:
                    break
                pairs = [(subwords[i], subwords[i+1]) for i in range(len(subwords) - 1)]
                if not any(pair in self.merges for pair in pairs):
                    break
                for i, pair in enumerate(pairs):
                    if pair in self.merges:
                        subwords[i] = self.merges[pair]
                        del subwords[i+1]
                        break
            tokens.extend(subwords)
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
        # Ensure we're saving the tokens exactly as they are, including spaces
        vocab_to_save = {repr(k): v for k, v in self.vocab.items()}
        save_vocab(vocab_to_save, vocab_file)
        print(f"\nVocabulary saved to {vocab_file}")

    def load(self, vocab_file):
        loaded_vocab = load_vocab(vocab_file)
        # Convert the loaded vocabulary back to the original format
        self.vocab = {eval(k): int(v) for k, v in loaded_vocab.items()}
        self.inverse_vocab = {int(v): eval(k) for k, v in loaded_vocab.items()}
