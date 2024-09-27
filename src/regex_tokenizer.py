import regex
from typing import List, Dict
from src.utils import save_vocab, load_vocab

class RegexTokenizer:
    PATTERNS: Dict[str, str] = {
        'basic': r'\b\w+\b|\S',
        'gpt2': r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        'gpt4': r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
        'improved': r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}+(?:[.,]\p{N}+)?| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    }

    def __init__(self, pattern: str = 'basic'):
        if pattern in self.PATTERNS:
            self.pattern = regex.compile(self.PATTERNS[pattern])
        else:
            self.pattern = regex.compile(pattern)
        self.vocab = {}
        self.inverse_vocab = {}

    def fit(self, text: str):
        """Build vocabulary from the given text."""
        tokens = self.tokenize(text)
        # Create a set of unique tokens, preserving leading spaces
        unique_tokens = set(tokens)
        # Create vocabulary, assigning each unique token an index
        self.vocab = {token: i for i, token in enumerate(unique_tokens)}
        self.inverse_vocab = {i: token for token, i in self.vocab.items()}

    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text using the specified regex pattern."""
        return self.pattern.findall(text)

    def encode(self, text: str) -> List[int]:
        """Encode the input text into token IDs."""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, len(self.vocab)) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Decode the token IDs back into text."""
        tokens = [self.inverse_vocab.get(id, '<unk>') for id in token_ids]
        return self.detokenize(tokens)

    def detokenize(self, tokens: List[str]) -> str:
        """Detokenize the input tokens by joining them."""
        return "".join(tokens)

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