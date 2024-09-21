# Tokenizer Tutorial: From Basic to Advanced

This repository contains implementations of various tokenization techniques, from simple whitespace tokenization to more advanced methods like Byte Pair Encoding (BPE). It serves as a practical guide to understanding and implementing different tokenization strategies used in Natural Language Processing (NLP).

## Tokenizers Included:

1. Whitespace Tokenizer
2. Basic Regex Tokenizer
3. Advanced Regex Tokenizer (with GPT-2 and GPT-4 inspired patterns)
4. Basic BPE (Byte Pair Encoding) Tokenizer
5. Advanced BPE Tokenizer (with pre-tokenization and post-processing)

## Getting Started

1. Clone this repository:
2. Install the required dependencies:
3. Explore the `src/` directory for tokenizer implementations.
4. Run examples from the `examples/` directory to see the tokenizers in action.
5. Check out the `tests/` directory for unit tests of each tokenizer.

## Usage

Each tokenizer can be used as follows:

```python
from src.whitespace_tokenizer import WhitespaceTokenizer

tokenizer = WhitespaceTokenizer()
tokenizer.train("Your training text goes here.")
encoded = tokenizer.encode("Text to encode")
decoded = tokenizer.decode(encoded)
```
