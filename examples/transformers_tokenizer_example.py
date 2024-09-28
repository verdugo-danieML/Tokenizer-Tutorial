from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer
from src.custom_hf_tokenizer import CustomHFTokenizer
from src.utils import read_corpus

def demonstrate_tokenizer(tokenizer, text):
    print(f"\nDemonstrating {tokenizer.__class__.__name__} functionality:")
    print("Input text:", text)

    if isinstance(tokenizer, CustomHFTokenizer):
        encoded = tokenizer.encode(text)
        tokens = tokenizer.tokenize(text)
        ids = encoded
    else:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)

    print("Tokenized:", tokens)
    print("Encoded:", ids)

    decoded = tokenizer.decode(ids)
    print("Decoded:", decoded)

def main():
    corpus_dir = "data/corpus/"
    text = read_corpus(corpus_dir)

    # Example text
    example_text = "This is an example of using custom tokenizers. It can handle punctuation, numbers like 3246, and contractions (e.g., don't, we'll)."

    # Custom HF Tokenizer
    custom_tokenizer = CustomHFTokenizer(vocab_size=10000, min_frequency=2)
    custom_tokenizer.train(corpus_dir)
    custom_tokenizer.save("trained_vocabs/hf_vocab.json")
    demonstrate_tokenizer(custom_tokenizer, example_text)

    # BERT Tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    demonstrate_tokenizer(bert_tokenizer, example_text)

    # GPT-2 Tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    demonstrate_tokenizer(gpt2_tokenizer, example_text)

    # RoBERTa Tokenizer
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    demonstrate_tokenizer(roberta_tokenizer, example_text)

if __name__ == "__main__":
    main()