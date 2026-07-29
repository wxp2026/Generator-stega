#!/usr/bin/env python3
"""
批量 DNA 隐写生成脚本

功能:
1. 从 DNA.txt 按行读取 DNA 序列
2. 读取 secret.txt 的二进制内容
3. 对每一条 DNA 序列执行隐写
4. 将每条隐写后的 DNA 按行保存到目标.txt

说明:
- 复用原 CCRS.py 中的核心隐写逻辑
- 默认对每条序列使用同一份 secret.txt 内容
- 每一行输出一条隐写 DNA 序列
"""

from pathlib import Path
from typing import List

from CCRS import encode_message


# =============================
# 配置区
# =============================
DNA_TXT = r".\5_PCA _ SWD\raw.txt"
SECRET_FILE = r".\4_Baselines\secret.txt"
OUTPUT_FILE = r".\7_Antisteganalysis Capabilities\Data\CCRS\steg.txt"
SECRET_KEY = r"my_secret_key_2024"
AUTO_OPTIMIZE = False
MIN_VALID_BASES = 3
# =============================


def load_secret_bytes(filepath: str) -> bytes:
    """读取 secret.txt 的原始二进制内容"""
    with open(filepath, "rb") as f:
        return f.read()


def normalize_dna_sequence(seq: str) -> str:
    """清洗 DNA 序列，只保留 A/C/G/T，并转为大写"""
    if seq is None:
        return ""
    return "".join(base for base in str(seq).upper() if base in "ACGT")


def load_dna_sequences_from_txt(txt_path: str) -> List[str]:
    """从 DNA.txt 中按行读取并清洗 DNA 序列"""
    sequences = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            seq = normalize_dna_sequence(line.strip())
            if seq:
                sequences.append(seq)
            else:
                print(f"[提示] 第 {line_num} 行为空或不含有效 A/C/G/T，已跳过")

    return sequences


def batch_encode_dna(
    dna_txt_path: str,
    secret_path: str,
    output_path: str,
    secret_key: str,
    auto_optimize: bool = True,
) -> None:
    """批量执行 DNA 隐写并保存到文本文件"""
    print("=" * 80)
    print("开始批量生成隐写 DNA")
    print("=" * 80)
    print(f"输入 DNA 文件: {dna_txt_path}")
    print(f"秘密文件: {secret_path}")
    print(f"输出文件: {output_path}")
    print(f"自动优化 fake ratio: {'是' if auto_optimize else '否'}")

    secret_bytes = load_secret_bytes(secret_path)
    print(f"已读取 secret.txt 二进制内容: {len(secret_bytes)} 字节")

    dna_sequences = load_dna_sequences_from_txt(dna_txt_path)
    print(f"从 DNA.txt 共读取到 {len(dna_sequences)} 条有效 DNA 序列")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count = 0
    fail_count = 0

    with open(output_file, "w", encoding="utf-8") as fout:
        for idx, reference_dna in enumerate(dna_sequences, start=1):
            try:
                if len(reference_dna) < MIN_VALID_BASES:
                    print(f"[跳过] 第 {idx} 条序列长度不足: {len(reference_dna)}")
                    skip_count += 1
                    continue

                synthesized_dna, metadata = encode_message(
                    message_data=secret_bytes,
                    key=secret_key,
                    reference_dna=reference_dna,
                    auto_optimize=auto_optimize,
                )

                fout.write(synthesized_dna + "\n")
                success_count += 1

                fake_ratio = metadata.get("fake_ratio", None)
                if fake_ratio is not None:
                    print(
                        f"[完成] 第 {idx} 条 | 原始长度={len(reference_dna)} | "
                        f"隐写长度={len(synthesized_dna)} | fake_ratio={fake_ratio:.2f}"
                    )
                else:
                    print(
                        f"[完成] 第 {idx} 条 | 原始长度={len(reference_dna)} | "
                        f"隐写长度={len(synthesized_dna)}"
                    )

            except Exception as e:
                fail_count += 1
                print(f"[失败] 第 {idx} 条处理出错: {e}")

    print("\n" + "=" * 80)
    print("批量处理结束")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")
    print(f"结果已保存到: {output_file.resolve()}")
    print("=" * 80)


def main() -> None:
    batch_encode_dna(
        dna_txt_path=DNA_TXT,
        secret_path=SECRET_FILE,
        output_path=OUTPUT_FILE,
        secret_key=SECRET_KEY,
        auto_optimize=AUTO_OPTIMIZE,
    )


if __name__ == "__main__":
    main()
