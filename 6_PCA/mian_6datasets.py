import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import cg_tm_kl


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ======================
# 基础参数
# ======================
SEQ_LENGTH = 198
KMER_LENGTH = 3
VECTOR_SIZE = 300
WINDOW = 3
MIN_COUNT = 1
EPOCHS = 10

# 输出目录
output_dir = r".\5_PCA _ SWD"
os.makedirs(output_dir, exist_ok=True)

# 模型路径
model_path = os.path.join(output_dir, "word2vec_3bp_6datasets.model")

# 数据处理参数
len_seq = 198
beg = 0
end = 5000
PADDING = False
flex = 10

# ======================
# 在这里配置 6 个数据集
# title: 图标题
# file_path: 数据文件路径
# tag: 输出文件名标记
# ======================
DATASETS = [
    {
        "title": "(a) raw",
        "file_path": r".\5_PCA _ SWD\dataset\raw_6.txt",
        "tag": "raw",
    },
    {
        "title": "(b) GCBS",
        "file_path": r".\5_PCA _ SWD\output\GCBS_clean.txt",
        "tag": "GCBS",
    },
    {
        "title": "(c) CCRS",
        "file_path": r".\5_PCA _ SWD\output\CCRS_clean.txt",
        "tag": "CCRS",
    },
    {
        "title": "(d) RBS",
        "file_path": r".\5_PCA _ SWD\dataset\RBS_clean.txt",
        "tag": "RBS",
    },
    {
        "title": "(e) LSTM-stega",
        "file_path": r".\5_PCA _ SWD\dataset\LSTM_stego_6_1.txt",
        "tag": "LSTM-stega",
    },
    {
        "title": "(f) Ours",
        "file_path": r".\5_PCA _ SWD\dataset\Ours_clean.txt",
        "tag": "Ours",
    },
]


# ======================
# 工具函数
# ======================
def split_words(line, num=3):
    return [line[i:i + num] for i in range(0, len(line), num) if len(line[i:i + num]) == num]



def clean_sequence(line):
    line = line.strip().replace(" ", "")
    return ''.join([ch for ch in line if ch in ['A', 'T', 'C', 'G']])



def read_and_split(file_path, kmer_length=3, seq_length=198):
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
    vectors = []
    for kmers in sequences:
        valid_vecs = [model.wv[kmer] for kmer in kmers if kmer in model.wv]
        if len(valid_vecs) == 0:
            vec = np.zeros(vector_size)
        else:
            vec = np.mean(valid_vecs, axis=0)
        vectors.append(vec)
    return np.array(vectors)



def train_or_load_word2vec(all_sequences, model_path):
    if os.path.exists(model_path):
        print(f"已加载模型：{model_path}")
        return Word2Vec.load(model_path)

    print("模型不存在，正在训练新模型...")
    print("训练样本数:", len(all_sequences))

    model = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=1,
        hs=1,
        epochs=EPOCHS,
    )

    model.build_vocab(all_sequences)
    print("词表大小:", len(model.wv))

    if len(model.wv) == 0:
        raise ValueError("词表为空，请检查输入数据或 k-mer 切分方式。")

    model.train(
        all_sequences,
        total_examples=len(all_sequences),
        epochs=model.epochs,
    )

    model.save(model_path)
    print(f"模型已保存：{model_path}")
    return model



def fit_joint_pca_multi(vectors_dict):
    """
    对多个数据集一起做 PCA，保证所有图都在同一坐标系下。
    vectors_dict: {dataset_tag: vectors_ndarray}
    return: {dataset_tag: points_2d}
    """
    tags = list(vectors_dict.keys())
    lengths = [len(vectors_dict[tag]) for tag in tags]
    all_vectors = np.vstack([vectors_dict[tag] for tag in tags])

    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_vectors)

    result = {}
    start = 0
    for tag, n in zip(tags, lengths):
        result[tag] = all_2d[start:start + n]
        start += n
    return result



def get_shared_axis_limits_multi(points_dict, margin_ratio=0.05):
    all_x = np.concatenate([points[:, 0] for points in points_dict.values()])
    all_y = np.concatenate([points[:, 1] for points in points_dict.values()])

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    x_margin = (x_max - x_min) * margin_ratio if x_max > x_min else 0.01
    y_margin = (y_max - y_min) * margin_ratio if y_max > y_min else 0.01

    xlim = (x_min - x_margin, x_max + x_margin)
    ylim = (y_min - y_margin, y_max + y_margin)
    return xlim, ylim



def plot_single_pca(points_2d, title, save_path, xlim=None, ylim=None, bins=220):
    x = points_2d[:, 0]
    y = points_2d[:, 1]

    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    x_idx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    y_idx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    density = hist[x_idx, y_idx]

    order = np.argsort(density)
    x_sorted = x[order]
    y_sorted = y[order]
    density_sorted = density[order]

    from matplotlib.colors import LinearSegmentedColormap
    density_cmap = LinearSegmentedColormap.from_list(
        'paper_density',
        ['#203a8c', '#2f6db3', '#47b8c8', '#f3e55d', '#f28e2b', '#c92525'],
        N=256,
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
        linewidths=0,
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



def compute_dataset_stats(file_path, seq_length=198):
    """
    计算单个数据集的 GC / Tm 均值。
    """
    lines = cg_tm_kl.txt_process_sc(file_path, seq_length, beg, end, PADDING, flex)

    cg_values = []
    tm_values = []

    for line in lines:
        line_temp = ''.join(line.strip().split(' '))
        cg_values.append(cg_tm_kl.C_G(line_temp))
        try:
            tm_values.append(cg_tm_kl.melting(line_temp))
        except Exception:
            continue

    cg_mean = np.mean(cg_values) if len(cg_values) > 0 else np.nan
    tm_mean = np.mean(tm_values) if len(tm_values) > 0 else np.nan
    return cg_mean, tm_mean


# ======================
# 主流程
# ======================
if __name__ == "__main__":
    if len(DATASETS) != 6:
        raise ValueError("DATASETS 必须正好配置 6 个数据集。")

    # 1) 检查文件是否存在
    for ds in DATASETS:
        if not os.path.exists(ds["file_path"]):
            raise FileNotFoundError(f"文件不存在: {ds['file_path']}")

    # 2) 读取 6 个数据集，并计算 GC/Tm
    dataset_sequences = {}
    all_sequences = []

    print("\n================ 数据集统计 ================")
    for ds in DATASETS:
        title = ds["title"]
        file_path = ds["file_path"]
        tag = ds["tag"]

        sequences = read_and_split(file_path, KMER_LENGTH, SEQ_LENGTH)
        dataset_sequences[tag] = sequences
        all_sequences.extend(sequences)

        cg_mean, tm_mean = compute_dataset_stats(file_path, SEQ_LENGTH)
        print(f"{title}")
        print(f"  文件: {file_path}")
        print(f"  序列数: {len(sequences)}")
        print(f"  CG mean: {cg_mean}")
        print(f"  Tm mean: {tm_mean}")
        if len(sequences) > 0:
            print(f"  示例前10个k-mer: {sequences[0][:10]}")
        print("-" * 50)

    # 3) 训练或加载统一 Word2Vec 模型
    word2vec_model = train_or_load_word2vec(all_sequences, model_path)

    # 4) 每个数据集转向量
    dataset_vectors = {}
    print("\n================ 向量维度 ================")
    for ds in DATASETS:
        tag = ds["tag"]
        vectors = sequences_to_vectors(dataset_sequences[tag], word2vec_model, VECTOR_SIZE)
        dataset_vectors[tag] = vectors
        print(f"{ds['title']}: {vectors.shape}")

    # 5) 6 个数据集联合 PCA，保证同一坐标系
    dataset_points_2d = fit_joint_pca_multi(dataset_vectors)

    # 6) 统一 6 张图坐标轴范围
    xlim, ylim = get_shared_axis_limits_multi(dataset_points_2d, margin_ratio=0.05)

    # 7) 循环绘制 6 张图
    print("\n================ 开始绘图 ================")
    for ds in DATASETS:
        tag = ds["tag"]
        title = ds["title"]
        save_path = os.path.join(output_dir, f"PCA_{tag}.jpg")

        plot_single_pca(
            points_2d=dataset_points_2d[tag],
            title=title,
            save_path=save_path,
            xlim=xlim,
            ylim=ylim,
        )

    print("\n全部 6 张 PCA 图已生成完成。")
