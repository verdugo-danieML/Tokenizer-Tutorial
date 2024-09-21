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
   ```bash
   git clone https://github.com/verdugo-danieML/Tokenizer-Tutorial.git
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Explore the `src/` directory for tokenizer implementations.
6. Run examples from the `examples/` directory to see the tokenizers in action.
7. Check out the `tests/` directory for unit tests of each tokenizer.

## Usage

Each tokenizer can be used as follows:

```python
from src.whitespace_tokenizer import WhitespaceTokenizer

tokenizer = WhitespaceTokenizer()
tokenizer.train("Your training text goes here.")
encoded = tokenizer.encode("Text to encode")
decoded = tokenizer.decode(encoded)
```
Replace WhitespaceTokenizer with the tokenizer of your choice.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

