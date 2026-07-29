import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cg_tm_kl
from matplotlib.colors import LinearSegmentedColormap
import random

import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 在主程序运行之前调用
set_seed(42)


# ======================
# 基础参数
# ======================
SEQ_LENGTH = 198       # 每条DNA序列长度
KMER_LENGTH = 3        # 3bp作为一个词
VECTOR_SIZE = 300
WINDOW = 3
MIN_COUNT = 1
EPOCHS = 10

# 数据路径

dp_or = r".\5_PCA _ SWD\raw_198.txt"
dp_sc = r".\5_PCA _ SWD\GCBS_198.txt"
# 输出目录
output_dir = r".\5_PCA _ SWD"
os.makedirs(output_dir, exist_ok=True)

# 模型路径
model_path = os.path.join(output_dir, "word2vec_3bp-1.model")

# 图像保存路径
original_pca_path = os.path.join(output_dir, "PCA_raw.jpg")
stego_pca_path = os.path.join(output_dir, "PCA_gcbs.jpg")

# 处理参数
len_ori = 198
len_sc = 198
beg_sc = 0
beg_ori = 0
end_sc = 5000
end_ori = 5000
PADDING = False
flex = 10


# ======================
# 工具函数
# ======================
def split_words(line, num=3):
    """
    将DNA序列按 num 长度切分
    例如 num=3 时，就是 3bp 一组
    """
    return [line[i:i + num] for i in range(0, len(line), num) if len(line[i:i + num]) == num]


def clean_sequence(line):
    """
    只保留 A/T/C/G，去掉空格、换行等
    """
    line = line.strip().replace(" ", "")
    return ''.join([ch for ch in line if ch in ['A', 'T', 'C', 'G']])


def read_and_split(file_path, kmer_length=3, seq_length=198):
    """
    从 txt 读取DNA序列，并按 3bp 切词
    """
    sequences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            seq = clean_sequence(line)
            seq = seq[:seq_length]

            if len(seq) == seq_length:
                kmers = split_words(seq, kmer_length)
                if len(kmers) > 0:
                    sequences.append(kmers)

    return sequences


def sequences_to_vectors(sequences, model, vector_size=300):
    """
    将k-mer序列转为向量表示：对所有词向量求平均
    """
    vectors = []

    for kmers in sequences:
        valid_vecs = []

        for kmer in kmers:
            if kmer in model.wv:
                valid_vecs.append(model.wv[kmer])

        if len(valid_vecs) == 0:
            vec = np.zeros(vector_size)
        else:
            vec = np.mean(valid_vecs, axis=0)

        vectors.append(vec)

    return np.array(vectors)


def train_or_load_word2vec(raw_sequences, steg_sequences, model_path):
    """
    如果模型存在则加载，否则训练新模型
    """
    if os.path.exists(model_path):
        print(f"已加载模型：{model_path}")
        return Word2Vec.load(model_path)

    print("模型不存在，正在训练新模型...")

    all_sequences = raw_sequences + steg_sequences
    print("训练样本数:", len(all_sequences))

    model = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=1,
        hs=1,
        epochs=EPOCHS
    )

    model.build_vocab(all_sequences)
    print("词表大小:", len(model.wv))

    if len(model.wv) == 0:
        raise ValueError("词表为空，请检查输入数据或k-mer切分方式。")

    model.train(
        all_sequences,
        total_examples=len(all_sequences),
        epochs=model.epochs
    )

    model.save(model_path)
    print(f"模型已保存：{model_path}")
    return model


def fit_joint_pca(original_vectors, stego_vectors):
    """
    原始序列和隐写序列一起做 PCA
    保证二者处于同一坐标系
    """
    pca = PCA(n_components=2)

    all_vectors = np.vstack([original_vectors, stego_vectors])
    all_2d = pca.fit_transform(all_vectors)

    n_ori = len(original_vectors)
    original_2d = all_2d[:n_ori]
    stego_2d = all_2d[n_ori:]

    return original_2d, stego_2d


def get_shared_axis_limits(original_2d, stego_2d, margin_ratio=0.05):
    """
    给两张图计算统一坐标范围
    """
    all_x = np.concatenate([original_2d[:, 0], stego_2d[:, 0]])
    all_y = np.concatenate([original_2d[:, 1], stego_2d[:, 1]])

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    x_margin = (x_max - x_min) * margin_ratio if x_max > x_min else 0.01
    y_margin = (y_max - y_min) * margin_ratio if y_max > y_min else 0.01

    xlim = (x_min - x_margin, x_max + x_margin)
    ylim = (y_min - y_margin, y_max + y_margin)

    return xlim, ylim


def plot_single_pca(points_2d, title, label, color, save_path, xlim=None, ylim=None, bins=220):
    """
    按局部密度给 PCA 散点上色：
    - 稀疏区域颜色更冷
    - 密集区域颜色更热、更亮
    - 对超多点场景比 gaussian_kde 更稳
    """
    x = points_2d[:, 0]
    y = points_2d[:, 1]

    # 用二维直方图估计每个点所在位置的局部密度
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    x_idx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    y_idx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    density = hist[x_idx, y_idx]

    # 先画稀疏点，再画密集点，让高密度区域更突出
    order = np.argsort(density)
    x_sorted = x[order]
    y_sorted = y[order]
    density_sorted = density[order]

    # 冷到热的密度渐变
    from matplotlib.colors import LinearSegmentedColormap
    density_cmap = LinearSegmentedColormap.from_list(
        'paper_density',
        ['#203a8c', '#2f6db3', '#47b8c8', '#f3e55d', '#f28e2b', '#c92525'],
        N=256
    )

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        x_sorted,
        y_sorted,
        c=density_sorted,
        cmap=density_cmap,
        marker='o',
        s=10,
        alpha=0.9,
        edgecolors='none',
        linewidths=0
    )

    cbar = plt.colorbar(sc, pad=0.02)
    cbar.set_label('Local point density', fontsize=11)

    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('PC1', fontsize=13)
    plt.ylabel('PC2', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.25)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()

    print(f"图像已保存：{save_path}")


# ======================
# 主流程
# ======================
if __name__ == "__main__":
    # 1) 按原项目方式处理数据，计算 CG 和 Tm
    line_sc = cg_tm_kl.txt_process_sc(dp_sc, len_sc, beg_sc, end_sc, PADDING, flex)
    line_ori = cg_tm_kl.txt_process_ori(dp_or, len_ori, beg_ori, end_ori, PADDING, flex)

    C_G_PER = []
    tm_PER = []

    for line in line_ori:
        line_temp = ''.join(line.strip().split(' '))
        C_G_PER.append(cg_tm_kl.C_G(line_temp))
        try:
            tm_PER.append(cg_tm_kl.melting(line_temp))
        except:
            continue

    C_G_PER_sc = []
    tm_PER_sc = []

    for line in line_sc:
        line_temp = ''.join(line.strip().split(' '))
        C_G_PER_sc.append(cg_tm_kl.C_G(line_temp))
        try:
            tm_PER_sc.append(cg_tm_kl.melting(line_temp))
        except:
            continue

    tm_mean_ori = np.mean(tm_PER)
    tm_mean_sc = np.mean(tm_PER_sc)
    C_G_PER_mean = np.mean(C_G_PER)
    C_G_PER_mean_SC = np.mean(C_G_PER_sc)

    print('tm_mean_ori:', tm_mean_ori)
    print('tm_mean_sc:', tm_mean_sc)
    print('C_G_MEAN_SC:', C_G_PER_mean_SC)
    print('C_G_MEAN_ORI:', C_G_PER_mean)
    print('tm_bias:', (np.abs(tm_mean_ori - tm_mean_sc) / tm_mean_ori) * 100, "%")
    print('CG_BIAS:', (np.abs(C_G_PER_mean_SC - C_G_PER_mean) / C_G_PER_mean) * 100, "%")
    print(cg_tm_kl.CG_b(line_ori, line_sc))
    print(cg_tm_kl.Tmb(line_ori, line_sc))

    # 2) 读取数据，并按 3bp 切词
    raw_sequences = read_and_split(dp_or, KMER_LENGTH, SEQ_LENGTH)
    stego_sequences = read_and_split(dp_sc, KMER_LENGTH, SEQ_LENGTH)

    print("raw_sequences 数量:", len(raw_sequences))
    print("stego_sequences 数量:", len(stego_sequences))

    if len(raw_sequences) > 0:
        print("raw 示例:", raw_sequences[0][:10])
    if len(stego_sequences) > 0:
        print("stego 示例:", stego_sequences[0][:10])

    # 3) 训练或加载 Word2Vec
    word2vec_model = train_or_load_word2vec(raw_sequences, stego_sequences, model_path)

    # 4) 序列转向量
    original_vectors = sequences_to_vectors(raw_sequences, word2vec_model, VECTOR_SIZE)
    stego_vectors = sequences_to_vectors(stego_sequences, word2vec_model, VECTOR_SIZE)

    print("original_vectors shape:", original_vectors.shape)
    print("stego_vectors shape:", stego_vectors.shape)

    # 5) PCA 降维（同一坐标系）
    original_2d, stego_2d = fit_joint_pca(original_vectors, stego_vectors)

    # 6) 统一两张图的坐标轴范围
    xlim, ylim = get_shared_axis_limits(original_2d, stego_2d, margin_ratio=0.05)

    # 7) 分别绘制两张图
    plot_single_pca(
        points_2d=original_2d,
        title='(a)original',
        label='(a)raw',
        color='#4C72B0',
        save_path=original_pca_path,
        xlim=xlim,
        ylim=ylim
    )

    plot_single_pca(
        points_2d=stego_2d,
        title='(f)GCBS',
        label='(f)GCBS',
        color='#DD8452',
        save_path=stego_pca_path,
        xlim=xlim,
        ylim=ylim
    )
