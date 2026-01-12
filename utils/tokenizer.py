import re
from collections import Counter

class SimpleTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        
    def normalize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def build_vocab(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(self.normalize(t).split())
        
        for word, _ in counter.most_common(self.vocab_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def encode(self, text, max_len=32):
        tokens = self.normalize(text).split()
        ids = [
            self.word2idx.get(tok, 1)
            for tok in tokens[:max_len]
        ]
        padding = [0] * (max_len - len(ids))
        return ids + padding