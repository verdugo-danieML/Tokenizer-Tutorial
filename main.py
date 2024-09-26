from src import (
    WhitespaceTokenizer, RegexTokenizer, BPETokenizer,
    CustomHFTokenizer, CustomSPTokenizer,
    read_corpus, get_vocab, save_vocab, load_vocab
)
import os
import argparse
import shutil

def get_vocab_file_with_extension(tokenizer_type, vocab_file):
    base_name = os.path.splitext(vocab_file)[0]
    if tokenizer_type in ["whitespace", "regex", "bpe"]:
        return f"{base_name}.txt"
    elif tokenizer_type == "sp":
        return f"{base_name}.model"
    elif tokenizer_type == "hf":
        return f"{base_name}.json"
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

def train_and_save_tokenizer(tokenizer, corpus_dir, vocab_file):
    corpus = read_corpus(corpus_dir)
    if isinstance(tokenizer, (BPETokenizer, CustomHFTokenizer, CustomSPTokenizer)):
        tokenizer.train(corpus_dir)
    else:
        tokenizer.fit(corpus)
    
    tokenizer.save(vocab_file)
    print(f"Tokenizer trained and vocabulary saved to {vocab_file}")

    # Clean up temporary files
    cleanup_temp_files()

def cleanup_temp_files():
    temp_files = ['spm_model.model', 'spm_model.vocab', 'temp_corpus.txt']
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted temporary file: {file}")

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
    parser.add_argument("--vocab_file", default="vocab", help="Vocabulary file name (without extension)")
    parser.add_argument("--sample_text", default="This is a sample text.", help="Sample text for tokenization")
    parser.add_argument("--train_file", help="Path to the file containing training data")
    
    args = parser.parse_args()

    tokenizer_map = {
        "whitespace": WhitespaceTokenizer,
        "regex": RegexTokenizer,
        "bpe": BPETokenizer,
        "hf": CustomHFTokenizer,
        "sp": CustomSPTokenizer
    }

    tokenizer_class = tokenizer_map[args.tokenizer]
    vocab_file = get_vocab_file_with_extension(args.tokenizer, args.vocab_file)
    
    if args.operation == "train":
        if not args.train_file:
            raise ValueError("--train_file must be specified when using the 'train' operation")
        
        tokenizer = tokenizer_class()
        train_and_save_tokenizer(tokenizer, args.train_file, vocab_file)

    elif args.operation == "use":
        if not os.path.exists(vocab_file):
            raise ValueError(f"Vocabulary file {vocab_file} does not exist. Train the tokenizer first.")
        
        load_and_use_tokenizer(tokenizer_class, vocab_file, args.sample_text)

    # Clean up temporary files after both train and use operations
    cleanup_temp_files()

if __name__ == "__main__":
    main()