from src.regex_tokenizer import RegexTokenizer
from src.utils import read_corpus


def demonstrate_tokenizer(tokenizer, text):
    print("\nDemonstrating Regex Tokenizer functionality:")
    print("Input text:", text)

    tokens = tokenizer.tokenize(text)
    print("Tokenized:", tokens)

    encoded = tokenizer.encode(text)
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

def main():
    corpus_dir = "data/corpus/"
    text = read_corpus(corpus_dir)
    
    # Train the tokenizer for each pattern  
    for pattern in RegexTokenizer.PATTERNS:
        # Initialize tokenizer
        tokenizer = RegexTokenizer(pattern)
        print("")
        print(f"Fitting Regex Tokenizer with pattern: {pattern}")
        tokenizer.fit(text)
        print(f"Regex Tokenizer fitted with pattern: {pattern}")
        print("Vocabulary size:", len(tokenizer.vocab))

        # Save the vocabulary
        vocab_file = f"trained_vocabs/regex_vocab_{pattern}.txt"
        tokenizer.save(vocab_file)

        # Load the vocabulary
        tokenizer.load(vocab_file)

        # Demonstrate tokenizer functionality
        text1 = "This is an example of the RegexTokenizer in action! It can handle punctuation, numbers like 3246, and contractions (e.g., don't, we'll)."

        demonstrate_tokenizer(tokenizer, text1)

if __name__ == "__main__":
    main()