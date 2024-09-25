from src.custom_sp_tokenizer import CustomSPTokenizer

def demonstrate_tokenizer(tokenizer, text):
    print("\nDemonstrating SentencePiece Tokenizer functionality:")
    print("Input text:", text)

    tokens = tokenizer.tokenize(text)
    print("Tokenized:", tokens)

    encoded = tokenizer.encode(text)
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

def main():
    corpus_dir = "data/corpus/"
    model_path = "trained_vocabs/spm_model.model"

    # Initialize and train the tokenizer
    tokenizer = CustomSPTokenizer(
        vocab_size=8000,
        model_type='unigram',  # You can also try 'bpe'
        character_coverage=0.9995,
        max_sentence_length=4192
    )
    print("Training SentencePiece Tokenizer...")
    tokenizer.train(corpus_dir)
    print("SentencePiece Tokenizer trained.")

    # Save the trained model
    tokenizer.save(model_path)
    print(f"Model saved to {model_path}")

    # Load the trained model
    new_tokenizer = CustomSPTokenizer()
    new_tokenizer.load(model_path)
    print(f"Model loaded from {model_path}")

    # Example text
    example_text = "This is an example of using SentencePiece tokenizer. It can handle punctuation, numbers like 3246, and contractions (e.g., don't, we'll)."

    demonstrate_tokenizer(new_tokenizer, example_text)

if __name__ == "__main__":
    main()