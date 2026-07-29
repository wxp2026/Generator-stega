from gensim.models import Word2Vec
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cg_tm_kl  # 确保 cg_tm_kl.py 在同目录

# 参数
seq_length = 198       # 每条序列长度
kmer_length = 6        # 你的数据已经是6bp一组
word2vec_model_file = "word2vec.model"  # 可以先训练或加载已有模型

# 读取文件并分割成k-mer
def read_and_split(file_path, kmer_length):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    sequences = []
    for line in lines:
        line = line.strip().replace(" ", "")  # 去掉空格
        # 分割成6bp一组
        kmers = [line[i:i+kmer_length] for i in range(0, len(line), kmer_length)]
        sequences.append(kmers)
    return sequences

# 训练Word2Vec模型（如果没有的话）
def train_word2vec(all_sequences):
    model = Word2Vec(all_sequences, vector_size=300, window=6, min_count=1, sg=1, hs=1, epochs=10)
    model.save(word2vec_model_file)
    return model

# 将序列转换成向量
def sequences_to_vectors(sequences, model):
    vectors = []
    for kmers in sequences:
        vec = np.zeros(300)
        for kmer in kmers:
            if kmer in model.wv:
                vec += model.wv[kmer]
        vectors.append(vec)
    return np.array(vectors)

# 绘制PCA图
def plot_pca(original_2D, steg_2D, save_path):
    plt.figure(figsize=(6,6))
    plt.scatter(original_2D[:,0], original_2D[:,1], c='blue', label='Original', alpha=0.6)
    plt.scatter(steg_2D[:,0], steg_2D[:,1], c='red', label='Stego', alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA: Original vs Stego")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

# ======================
# 主流程
# ======================
# 读取序列
raw_sequences = read_and_split("dataset/raw.txt", kmer_length)
steg_sequences = read_and_split("steg.txt", kmer_length)

# 训练或加载Word2Vec
try:
    w2v_model = Word2Vec.load(word2vec_model_file)
    print("已加载已有模型")
except:
    print("训练新模型")
    w2v_model = train_word2vec(raw_sequences + steg_sequences)

# 转向量
raw_vectors = sequences_to_vectors(raw_sequences, w2v_model)
steg_vectors = sequences_to_vectors(steg_sequences, w2v_model)

# PCA降维
raw_2D = PCA(n_components=2).fit_transform(raw_vectors)
steg_2D = PCA(n_components=2).fit_transform(steg_vectors)

# 绘图
plot_pca(raw_2D, steg_2D, "PCA_raw_vs_steg.jpg")
