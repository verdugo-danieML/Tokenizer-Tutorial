from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders, trainers
from src.utils import read_corpus

class CustomHFTokenizer:
    def __init__(self, vocab_size=25000):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
            show_progress=True
        )
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        self.tokenizer.decoder = decoders.ByteLevel()

    def train(self, corpus_dir):
        corpus = read_corpus(corpus_dir)
        self.tokenizer.train_from_iterator([corpus], trainer=self.trainer)

    def save(self, path):
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens