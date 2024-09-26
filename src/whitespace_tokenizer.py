from typing import List, Dict
from src.utils import save_vocab, load_vocab

class WhitespaceTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}

    def fit(self, text: str):
        """Build vocabulary from the given text."""
        tokens = self.tokenize(text)
        self.vocab = {token: i for i, token in enumerate(set(tokens))}
        self.inverse_vocab = {i: token for token, i in self.vocab.items()}

    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text using whitespace as delimiter."""
        return text.split()

    def encode(self, text: str) -> List[int]:
        """Encode the input text into token IDs."""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, len(self.vocab)) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Decode the token IDs back into text."""
        tokens = [self.inverse_vocab.get(id, '<unk>') for id in token_ids]
        return self.detokenize(tokens)

    def detokenize(self, tokens: List[str]) -> str:
        """Detokenize the input tokens by joining them with a space."""
        return " ".join(tokens)
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