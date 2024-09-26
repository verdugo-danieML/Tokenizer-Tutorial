# Tokenizer Tutorial: From Basic to Advanced

This project implements various tokenization techniques from scratch, including Whitespace, Regex, Byte-Pair Encoding (BPE), and integrates with Hugging Face and SentencePiece tokenizers. It also includes a web application for visualizing and comparing different tokenization methods.

## Project Structure

```
Tokenizer-Tutorial/
│
├── data/
│ └── corpus/
│ ├── text_1.txt
│ ├── text_2.txt
│ └── text_3.txt
│
├── src/
│ ├── init.py
│ ├── utils.py
│ ├── whitespace_tokenizer.py
│ ├── regex_tokenizer.py
│ ├── bpe_tokenizer.py
│ ├── custom_hf_tokenizer.py
│ └── custom_sp_tokenizer.py
│
├── examples/
│ ├── whitespace_tokenizer_example.py
│ ├── regex_tokenizer_example.py
│ ├── bpe_tokenizer_example.py
│ ├── transformers_tokenizer_example.py
│ └── sentencepiece_tokenizer_example.py
│
├── templates/
│ └── index.html
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/verdugo-danieML/Tokenizer-Tutorial.git
   cd Tokenizer-Tutorial
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

You can use the `main.py` script to train and use different tokenizers:

```
python main.py <tokenizer_type> <operation> [--vocab_file VOCAB_FILE] [--sample_text SAMPLE_TEXT] [--train_file TRAIN_FILE]
```
- `<tokenizer_type>`: Choose from `whitespace`, `regex`, `bpe`, `hf` (Hugging Face), or `sp` (SentencePiece).
- `<operation>`: Choose `train` to train a new tokenizer or `use` to use a pre-trained tokenizer.
- `--vocab_file`: Specify the path to save/load the vocabulary file (default: vocab.txt).
- `--sample_text`: Provide a sample text for tokenization when using a pre-trained tokenizer.
- `--train_file`: Specify the path to the file containing training data when training a new tokenizer.

### Example usage:

1. Training a tokenizer:
```
 python main.py bpe train --vocab_file trained_vocabs/bpe_vocab.txt --train_file data/corpus/
```
use:
```
 python main.py bpe use --vocab_file trained_vocabs/bpe_vocab.txt --sample_text "This is a test sentence."
```

You can also run individual example scripts to see how each tokenizer works:

```
python -m examples.whitespace_tokenizer_example
python -m examples.regex_tokenizer_example
python -m examples.bpe_tokenizer_example
python -m examples.transformers_tokenizer_example
python -m examples.sentencepiece_tokenizer_example
```
## Web Application

To run the web app for visualizing tokenizers:

1. Ensure you have installed all required dependencies.

2. Run the Flask app:
   ```
   python app.py
   ```

3. Open a web browser and go to `http://localhost:5000`

4. Use the web interface to:
   - Tokenize text using multiple tokenizers simultaneously
   - Compare tokenization results side-by-side
   - View token frequencies
   - Train tokenizers on custom corpora by uploading text files

## Tokenizer Types

1. **Whitespace Tokenizer**: Splits text on whitespace.
2. **Regex Tokenizer**: Uses regular expressions for flexible tokenization.
3. **BPE (Byte-Pair Encoding) Tokenizer**: Implements the BPE algorithm for subword tokenization.
4. **Custom Hugging Face Tokenizer**: Integrates with the Hugging Face tokenizers library.
5. **Custom SentencePiece Tokenizer**: Integrates with the SentencePiece library.

## Contributing

Feel free to submit issues or pull requests if you have suggestions for improvements or find any bugs.

## License

This project is open-source and available under the MIT License.
