import math
import collections
import os

# ==================== 配置区域 ====================
# 1. 原始参考序列 (只有几行的那个文件)
FILE_ORI = r'.\3_Model_training & Stega_generating\adg_train\data\positive_samples.txt'

# 2. 隐写生成的序列 (生成的 dna.txt)
FILE_STEGO = r'.\3_Model_training & Stega_generating\adg_train\data\negative data'


# =================================================

def read_seq(path):
    """读取并合并为一个长字符串"""
    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}")
        return ""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        with open(path, 'r', encoding='gbk') as f:
            lines = f.readlines()

    # 清洗：只保留ATCG，变成一条长链
    seq = "".join([l.strip().upper() for l in lines])
    import re
    seq = re.sub(r'[^ATCG]', '', seq)
    return seq


def get_kmer_counts(seq, k=2):
    """统计 k-mer 频次"""
    counts = collections.defaultdict(int)
    # 初始化所有可能的组合，确保没有0 (平滑处理的关键)
    bases = ['A', 'T', 'C', 'G']
    if k == 1:
        permutations = bases
    elif k == 2:
        permutations = [b1 + b2 for b1 in bases for b2 in bases]

    # 拉普拉斯平滑：给每个组合预设 1 次计数
    for p in permutations:
        counts[p] = 1

    # 实际统计
    for i in range(len(seq) - k + 1):
        sub = seq[i:i + k]
        counts[sub] += 1

    return counts, sum(counts.values())


def calc_metrics(seq_ori, seq_stego):
    if not seq_ori or not seq_stego:
        return

    print(f"原始链长度: {len(seq_ori)} bp")
    print(f"隐写链长度: {len(seq_stego)} bp")

    # ---------------------------
    # 1. 计算 GC 含量差异
    # ---------------------------
    def get_gc(s):
        return (s.count('G') + s.count('C')) / len(s) * 100

    gc_ori = get_gc(seq_ori)
    gc_stego = get_gc(seq_stego)
    gc_diff = abs(gc_ori - gc_stego)

    # ---------------------------
    # 2. 计算 Tm (熔解温度) 差异
    # ---------------------------
    # 简易 Tm 公式: Tm = 64.9 + 41*(G+C-16.4)/(L)  (适用于长序列近似)
    # 对于短序列通常用 Wallace rule，这里用论文常用的长序列公式近似
    def get_tm(s):
        # 避免长度为0
        if len(s) == 0: return 0
        gc_count = s.count('G') + s.count('C')
        return 64.9 + 41 * (gc_count - 16.4) / len(s)

    tm_ori = get_tm(seq_ori)
    tm_stego = get_tm(seq_stego)
    tm_diff = abs(tm_ori - tm_stego)

    # ---------------------------
    # 3. 计算 KL 散度 (基于 Dinucleotide)
    # ---------------------------
    # 统计双碱基 (AA, AT...)
    counts_p, total_p = get_kmer_counts(seq_stego, k=2)  # P = Stego
    counts_q, total_q = get_kmer_counts(seq_ori, k=2)  # Q = Original (Reference)

    kl_sum = 0.0
    # 遍历所有 16 种组合
    permutations = [b1 + b2 for b1 in ['A', 'T', 'C', 'G'] for b2 in ['A', 'T', 'C', 'G']]

    for kmer in permutations:
        p_x = counts_p[kmer] / total_p
        q_x = counts_q[kmer] / total_q

        # KL 公式: Sum( P(x) * log2( P(x) / Q(x) ) )
        # 因为做了平滑，q_x 绝不会是 0
        kl_sum += p_x * math.log2(p_x / q_x)

    # ---------------------------
    # 输出结果
    # ---------------------------
    print("\n" + "=" * 40)
    print("🔬 单对单序列对比结果")
    print("=" * 40)
    print(f"1. GC Content Bias (GC偏差):  {gc_diff:.4f} %")
    print(f"   (Ori: {gc_ori:.2f}%, Stego: {gc_stego:.2f}%)")
    print("-" * 40)
    print(f"2. Tm Bias (熔解温度偏差):    {tm_diff:.4f}")
    print(f"   (Ori: {tm_ori:.2f}, Stego: {tm_stego:.2f})")
    print("-" * 40)
    print(f"3. KL Divergence (KL散度):    {kl_sum:.6f}")
    print("   (基于二核苷酸频率，已做拉普拉斯平滑)")
    print("=" * 40)


if __name__ == "__main__":
    s_ori = read_seq(FILE_ORI)
    s_stego = read_seq(FILE_STEGO)
    calc_metrics(s_ori, s_stego)