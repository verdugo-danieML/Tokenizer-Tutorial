from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from src.utils import read_corpus

class CustomHFTokenizer:
    def __init__(self, vocab_size=30000, min_frequency=2):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.trainer = BpeTrainer(
            vocab_size=vocab_size, 
            min_frequency=min_frequency,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )
        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )

    def train(self, corpus_dir):
        corpus = read_corpus(corpus_dir)
        self.tokenizer.train_from_iterator([corpus], trainer=self.trainer)

    def save(self, path):
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)