import os
from typing import List

from GCBS import embed_message

# ==================== 配置区 ====================
DNA_INPUT_FILE = r".\7_Antisteganalysis Capabilities\Data\RBS\raw_clean.txt"  # 输入DNA序列文件（每行一个）
SECRET_FILE = r".\4_Baselines\half secret.txt"          # 秘密信息
OUTPUT_FILE = r".\7_Antisteganalysis Capabilities\GCBS.txt"    # 输出隐写后的DNA
SECRET_KEY = r"EGYPT"
MIN_VALID_BASES = 3
# ==============================================


def clean_dna_sequence(seq: str) -> str:
    """清洗DNA序列，只保留ACGT"""
    if seq is None:
        return ""
    return "".join(base for base in str(seq).upper() if base in "ACGT")


def ensure_output_dir(path: str) -> None:
    """如果输出目录不存在则自动创建"""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def read_secret_as_text_from_binary(secret_path: str) -> str:
    """
    以二进制方式读取secret.txt
    再使用latin-1映射为字符串，保证字节不丢失
    """
    with open(secret_path, "rb") as f:
        secret_bytes = f.read()

    if not secret_bytes:
        raise ValueError("secret.txt 为空，无法进行隐写。")

    return secret_bytes.decode("latin-1")


def load_dna_sequences_from_txt(txt_path: str) -> List[str]:
    """
    从DNA.txt读取DNA序列
    每行一个DNA
    """
    sequences = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            seq = clean_dna_sequence(line.strip())
            if len(seq) >= MIN_VALID_BASES:
                sequences.append(seq)

    return sequences


def calc_embedded_bits(secret_message: str) -> int:
    """计算秘密信息总比特数。latin-1 下 1 字符 = 1 字节。"""
    return len(secret_message.encode("latin-1")) * 8


def calc_bpn(embedded_bits: int, dna_length: int) -> float:
    """计算 BPN（bits per nucleotide）。"""
    if dna_length <= 0:
        return 0.0
    return embedded_bits / dna_length


def main() -> None:

    print("=" * 70)
    print("GCBS 批量DNA隐写生成")
    print("=" * 70)
    print(f"输入DNA文件: {DNA_INPUT_FILE}")
    print(f"秘密文件: {SECRET_FILE}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"密钥: {SECRET_KEY}")
    print()

    if not os.path.exists(DNA_INPUT_FILE):
        raise FileNotFoundError(f"未找到DNA.txt文件: {DNA_INPUT_FILE}")

    if not os.path.exists(SECRET_FILE):
        raise FileNotFoundError(f"未找到secret.txt文件: {SECRET_FILE}")

    # 读取DNA序列
    dna_sequences = load_dna_sequences_from_txt(DNA_INPUT_FILE)

    if not dna_sequences:
        raise ValueError("DNA.txt 中未读取到有效DNA序列。")

    # 读取秘密信息
    secret_message = read_secret_as_text_from_binary(SECRET_FILE)
    embedded_bits = calc_embedded_bits(secret_message)

    ensure_output_dir(OUTPUT_FILE)

    success_count = 0
    fail_count = 0
    total_embedded_bits = 0
    total_stego_nt = 0

    print(f"秘密信息总长度: {len(secret_message.encode('latin-1'))} 字节 ({embedded_bits} bits)")
    print()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:

        for idx, dna_seq in enumerate(dna_sequences, start=1):

            try:
                stego_dna = embed_message(dna_seq, secret_message, SECRET_KEY)
                stego_len = len(stego_dna)
                bpn = calc_bpn(embedded_bits, stego_len)

                out_f.write(stego_dna + "\n")

                success_count += 1
                total_embedded_bits += embedded_bits
                total_stego_nt += stego_len

                print(
                    f"[{idx}/{len(dna_sequences)}] 成功，"
                    f"原长度={len(dna_seq)}，"
                    f"隐写后长度={stego_len}，"
                    f"容量={bpn:.6f} BPN"
                )

            except Exception as e:

                fail_count += 1
                print(f"[{idx}/{len(dna_sequences)}] 失败: {e}")
                continue

    avg_bpn = calc_bpn(total_embedded_bits, total_stego_nt)

    print()
    print("=" * 70)
    print("批量生成完成")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总嵌入比特数: {total_embedded_bits} bits")
    print(f"总隐写序列长度: {total_stego_nt} nt")
    print(f"平均隐写容量: {avg_bpn:.6f} BPN")
    print(f"结果已保存到: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
