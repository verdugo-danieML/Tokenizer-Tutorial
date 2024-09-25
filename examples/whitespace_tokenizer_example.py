from src.whitespace_tokenizer import WhitespaceTokenizer
from src.utils import read_corpus

def demonstrate_tokenizer(tokenizer, text):
    print("\nDemonstrating Whitespace Tokenizer functionality:")
    print("Input text:", text)

    tokens = tokenizer.tokenize(text)
    print("Tokenized:", tokens)

    encoded = tokenizer.encode(text)
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

def main():
    corpus_dir = "data/corpus/"
    vocab_file = "trained_vocabs/whitespace_vocab.txt"
    
    text = read_corpus(corpus_dir)

    # Initialize tokenizer
    tokenizer = WhitespaceTokenizer()

    # Train the tokenizer
    print("Fitting Whitespace Tokenizer...")
    tokenizer.fit(text)
    print("Whitespace Tokenizer fitted.")
    print("Vocabulary size:", len(tokenizer.vocab))

    # Save the vocabulary
    tokenizer.save(vocab_file)

    # Load the vocabulary
    tokenizer.load(vocab_file)

    # Demonstrate tokenizer functionality
    text1 = "This is a test sentence with some unknown words like xylophone."
    text2 = "The quick brown fox jumps over the lazy dog."

    demonstrate_tokenizer(tokenizer, text1)

if __name__ == "__main__":
    main()