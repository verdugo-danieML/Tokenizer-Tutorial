from src.bpe_tokenizer import BPETokenizer

def demonstrate_tokenizer(tokenizer, text):
    print("\nDemonstrating BPE Tokenizer functionality:")
    print("Input text:", text)

    tokens = tokenizer.tokenize(text)
    print("Tokenized:", tokens)

    encoded = tokenizer.encode(text)
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

def main():
    corpus_dir = "data/corpus/"
    vocab_file = "trained_vocabs/bpe_vocab.txt"

    # Train and save
    tokenizer = BPETokenizer()
    tokenizer.train(corpus_dir)
    tokenizer.save(vocab_file)

    # Create a new tokenizer instance and load
    new_tokenizer = BPETokenizer()
    new_tokenizer.load(vocab_file)

    # Demonstrate tokenizer functionality
    text = "This is a test sentence with some unknown words like xylophone."
    demonstrate_tokenizer(new_tokenizer, text)

if __name__ == "__main__":
    main()