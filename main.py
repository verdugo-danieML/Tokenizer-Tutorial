from src import (
    WhitespaceTokenizer, RegexTokenizer, BPETokenizer,
    CustomHFTokenizer, CustomSPTokenizer,
    read_corpus, get_vocab, save_vocab, load_vocab
)
import os
import argparse

def train_and_save_tokenizer(tokenizer, corpus_dir, vocab_file):
    corpus = read_corpus(corpus_dir)
    if isinstance(tokenizer, (BPETokenizer, CustomHFTokenizer, CustomSPTokenizer)):
        tokenizer.train(corpus_dir)
    else:
        tokenizer.fit(corpus)
    tokenizer.save(vocab_file)
    print(f"Tokenizer trained and vocabulary saved to {vocab_file}")

def load_and_use_tokenizer(tokenizer_class, vocab_file, sample_text):
    tokenizer = tokenizer_class()
    tokenizer.load(vocab_file)
    
    tokens = tokenizer.tokenize(sample_text)
    print(f"Tokenized: {tokens}")
    encoded = tokenizer.encode(sample_text)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

def main():
    parser = argparse.ArgumentParser(description="Tokenizer operations")
    parser.add_argument("tokenizer", choices=["whitespace", "regex", "bpe", "hf", "sp"], help="Tokenizer to use")
    parser.add_argument("operation", choices=["train", "use"], help="Operation to perform")
    parser.add_argument("--vocab_file", default="vocab.txt", help="Vocabulary file name")
    parser.add_argument("--sample_text", default="This is a sample text.", help="Sample text for tokenization")
    args = parser.parse_args()

    tokenizer_map = {
        "whitespace": WhitespaceTokenizer,
        "regex": RegexTokenizer,
        "bpe": BPETokenizer,
        "hf": CustomHFTokenizer,
        "sp": CustomSPTokenizer
    }

    tokenizer_class = tokenizer_map[args.tokenizer]
    
    if args.operation == "train":
        corpus_dir = os.path.join("data", "corpus")
        tokenizer = tokenizer_class()
        train_and_save_tokenizer(tokenizer, corpus_dir, args.vocab_file)
    elif args.operation == "use":
        load_and_use_tokenizer(tokenizer_class, args.vocab_file, args.sample_text)

if __name__ == "__main__":
    main()
