import os
import numpy as np
import matplotlib.pyplot as pl
import ot


# 读取 DNA 文件并将每行的 DNA 序列保存为一个列表
def read_dna_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        dna_sequences = f.readlines()
    # 去掉每行末尾的换行符，并去除空格
    return [''.join(seq.strip().split()) for seq in dna_sequences if seq.strip()]


# 将 DNA 序列转换为数值特征向量
def dna_to_numeric(dna_sequence):
    """将 DNA 序列转换为数值向量"""
    dna_to_num = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    # 只处理 A/T/C/G，其他字符记为 -1
    return [dna_to_num[base] if base in dna_to_num else -1 for base in dna_sequence]


# 计算切片 Wasserstein 距离 (SWD)
def getEMD(xs, xt, num_samples, path):
    # 样本数必须一致，否则 a/b 和数据长度会不匹配
    n = min(len(xs), len(xt), num_samples)
    xs = xs[:n]
    xt = xt[:n]

    a = np.ones((n,)) / n
    b = np.ones((n,)) / n

    # 创建图片输出目录
    save_dir = 'PICC'
    os.makedirs(save_dir, exist_ok=True)

    # 生成安全的保存文件名
    # path 只是一个名字来源，不再用字符串切片硬砍
    savename = os.path.basename(path)
    if not savename:
        savename = 'emd_result'

    # ---------- 图1：源分布和目标分布 ----------
    # 你的 DNA 数值向量通常是高维的，不能直接默认用 xs[:,0], xs[:,1] 画，
    # 但这里至少要求二维以上
    if xs.ndim != 2 or xt.ndim != 2:
        raise ValueError("xs 和 xt 必须是二维数组，形状应为 (样本数, 特征维度)")

    if xs.shape[1] < 2 or xt.shape[1] < 2:
        raise ValueError("每个样本至少需要 2 维特征，才能绘制二维散点图")

    pl.figure(figsize=(8, 6))
    pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    pl.legend(loc=0)
    pl.title('Source and target distributions')

    fig1_path = os.path.join(save_dir, savename + '_distribution.jpg')
    pl.savefig(fig1_path, format='jpg', dpi=300, bbox_inches='tight')
    pl.close()

    # ---------- 计算 SWD ----------
    n_seed = 50
    n_projections_arr = np.logspace(0, 3, 25, dtype=int)
    res = np.empty((n_seed, len(n_projections_arr)))

    for seed in range(n_seed):
        for i, n_projections in enumerate(n_projections_arr):
            res[seed, i] = ot.sliced_wasserstein_distance(
                xs, xt, a, b, n_projections, seed=seed
            )

    # 计算均值和标准差
    res_mean = np.mean(res, axis=0)
    res_std = np.std(res, axis=0)

    # ---------- 图2：SWD 曲线 ----------
    pl.figure(figsize=(8, 6))
    pl.plot(n_projections_arr, res_mean, label="SWD")
    pl.fill_between(
        n_projections_arr,
        res_mean - 2 * res_std,
        res_mean + 2 * res_std,
        alpha=0.5
    )

    pl.legend()
    pl.xscale('log')
    pl.xlabel("Number of projections")
    pl.ylabel("Distance")
    pl.title('Sliced Wasserstein Distance with 95% confidence interval')

    fig2_path = os.path.join(save_dir, savename + '_swd.jpg')
    pl.savefig(fig2_path, format='jpg', dpi=300, bbox_inches='tight')
    pl.close()

    print("图像已保存到：")
    print(fig1_path)
    print(fig2_path)

    return res_mean[-1]


# 主函数
if __name__ == '__main__':
    # 读取文件
    file1_path = r'.\5_PCA _ SWD\raw_198.txt'
    file2_path = r'.\5_PCA _ SWD\output\Ours_clean.txt'

    file1_dna = read_dna_file(file1_path)
    file2_dna = read_dna_file(file2_path)

    # 转成数值特征
    xs = np.array([dna_to_numeric(seq) for seq in file1_dna], dtype=float)
    xt = np.array([dna_to_numeric(seq) for seq in file2_dna], dtype=float)

    # 检查序列长度是否一致
    if xs.shape[1] != xt.shape[1]:
        raise ValueError(f"两个文件中的 DNA 序列特征维度不一致: xs.shape={xs.shape}, xt.shape={xt.shape}")

    num_samples = min(len(xs), len(xt))

    # 这里只作为保存名来源，不再当真实文件路径使用
    path = r'SWD'

    result = getEMD(xs, xt, num_samples=num_samples, path=path)
    print("最终 SWD =", result)
