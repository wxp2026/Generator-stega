# utils.py
import collections
import numpy as np
import os


class Vocabulary(object):
    def __init__(self, data_path, max_len=200, min_len=5, word_drop=0):
        if isinstance(data_path, str):
            data_path = [data_path]
        self._data_path = data_path
        self.w2i = {'_PAD': 0, '_UNK': 1, '_BOS': 2, '_EOS': 3}
        self.i2w = {0: '_PAD', 1: '_UNK', 2: '_BOS', 3: '_EOS'}
        self.max_len = max_len
        self.min_len = min_len
        self._build_vocabulary()

    def _build_vocabulary(self):
        words_all = []
        for path in self._data_path:
            if not os.path.exists(path): continue
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = line.strip().split()
                    if self.min_len <= len(tokens) <= self.max_len:
                        words_all.extend(tokens)

        counter = collections.Counter(words_all)
        sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        for word, _ in sorted_words:
            if word not in self.w2i:
                idx = len(self.w2i)
                self.w2i[word] = idx
                self.i2w[idx] = word
        self.vocab_size = len(self.w2i)


class Corpus(object):
    def __init__(self, data_path, vocabulary, max_len=200, min_len=5):
        self.corpus = []
        self.vocab = vocabulary
        if isinstance(data_path, str): data_path = [data_path]

        for path in data_path:
            if not os.path.exists(path): continue
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = line.strip().split()
                    if min_len <= len(tokens) <= max_len:
                        # 添加 BOS 和 EOS
                        ids = [self.vocab.w2i.get(t, self.vocab.w2i['_UNK']) for t in tokens]
                        self.corpus.append([self.vocab.w2i['_BOS']] + ids + [self.vocab.w2i['_EOS']])


class Generator(object):
    def __init__(self, data, vocabulary):
        self.data = np.array(data, dtype=object)
        self.vocab = vocabulary

    def get_batch(self, batch_size):
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)

        for start in range(0, len(self.data), batch_size):
            end = min(start + batch_size, len(self.data))
            batch_idx = indices[start:end]
            batch_seqs = self.data[batch_idx]

            # Padding
            max_len = max(len(s) for s in batch_seqs)
            padded_batch = np.zeros((len(batch_seqs), max_len), dtype=np.int64)
            padded_batch.fill(self.vocab.w2i['_PAD'])

            for i, seq in enumerate(batch_seqs):
                padded_batch[i, :len(seq)] = seq

            yield padded_batch


def create_dummy_data(path):
    """如果数据不存在，生成模拟的DNA数据"""
    bases = ['A', 'T', 'C', 'G']
    import random
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        for _ in range(500):  # 生成500条数据
            length = random.randint(10, 50)
            # 模拟按照 codon 切分 (例如每3个碱基为一个词，或者单碱基)
            # 这里为了简单，假设单碱基或者是2-mer
            seq = []
            for _ in range(length):
                seq.append("".join(random.choices(bases, k=2)))  # 2-mer base unit
            f.write(" ".join(seq) + "\n")
    print(f"Created dummy data at {path}")