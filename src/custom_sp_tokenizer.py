import sentencepiece as spm
from src.utils import read_corpus
import os

class CustomSPTokenizer:
    def __init__(self, vocab_size=8000, model_type='unigram', character_coverage=0.9995, max_sentence_length=4192):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.max_sentence_length = max_sentence_length
        self.sp = None
        self.model_prefix = 'spm_model'

    def train(self, corpus_dir):
        text = read_corpus(corpus_dir)
        with open('temp_corpus.txt', 'w', encoding='utf-8') as f:
            f.write(text)

        spm.SentencePieceTrainer.train(
            input='temp_corpus.txt',
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            character_coverage=self.character_coverage,
            max_sentence_length=self.max_sentence_length,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='[PAD]',
            unk_piece='[UNK]',
            bos_piece='[BOS]',
            eos_piece='[EOS]'
        )

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f'{self.model_prefix}.model')

    def save(self, path):
        if self.sp:
            # Copy the trained model file to the specified path
            import shutil
            shutil.copy(f'{self.model_prefix}.model', path)
            print(f"Model saved to {path}")
        else:
            print("No model to save. Train the tokenizer first.")

    def load(self, path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(path)

    def encode(self, text):
        return self.sp.encode_as_ids(text)

    def decode(self, ids):
        return self.sp.decode_ids(ids)

    def tokenize(self, text):
        return self.sp.encode_as_pieces(text)