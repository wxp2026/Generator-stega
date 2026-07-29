#!/usr/bin/env python3
"""
批量 DNA 隐写生成脚本（含整体隐写容量 BPN 统计）

功能:
1. 从 DNA.txt 按行读取 DNA 序列
2. 读取 secret.txt 的二进制内容
3. 对每一条 DNA 序列执行隐写
4. 将每条隐写后的 DNA 按行保存到目标.txt
5. 输出每条序列及整体的隐写容量 BPN（bits per nucleotide）
6. 将统计明细保存为 CSV 文件

说明:
- 复用原 CCRS.py 中的核心隐写逻辑
- 默认对每条序列使用同一份 secret.txt 内容
- 优先从 encode_message 返回的 metadata 中提取“真实嵌入比特数”
- 若 metadata 中未提供对应字段，则回退为 secret.txt 的总 bit 数做估算
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from CCRS import encode_message


# =============================
# 配置区
# =============================
DNA_TXT = r".\7_Antisteganalysis Capabilities\Data\RBS\raw_clean.txt"
SECRET_FILE = r".\4_Baselines\secret.txt"
OUTPUT_FILE = r".\7_Antisteganalysis Capabilities\CCRS.txt"
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
    sequences: List[str] = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            seq = normalize_dna_sequence(line.strip())
            if seq:
                sequences.append(seq)
            else:
                print(f"[提示] 第 {line_num} 行为空或不含有效 A/C/G/T，已跳过")

    return sequences


def ensure_metadata_dict(metadata: Any) -> Dict[str, Any]:
    """尽量把 metadata 规整成 dict，避免因返回类型不同报错"""
    if isinstance(metadata, dict):
        return metadata
    if metadata is None:
        return {}
    try:
        return dict(metadata)
    except Exception:
        return {}


def try_parse_number(value: Any) -> Optional[float]:
    """将 metadata 中可能出现的数字安全转为 float"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
            return float(text)
    return None


def first_numeric_value(metadata: Dict[str, Any], keys: List[str]) -> Tuple[Optional[float], Optional[str]]:
    """按候选字段顺序提取数值"""
    for key in keys:
        if key in metadata:
            number = try_parse_number(metadata.get(key))
            if number is not None:
                return number, key
    return None, None


def first_bitstring_length(metadata: Dict[str, Any], keys: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """从 metadata 中的 01 串长度推断嵌入 bit 数"""
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text and set(text) <= {"0", "1"}:
                return len(text), key
    return None, None


def infer_embedded_bits(
    metadata: Dict[str, Any],
    secret_bytes: bytes,
    synthesized_dna: str,
) -> Tuple[int, str, str]:
    """
    推断本条样本真实嵌入的 bit 数。

    返回:
        embedded_bits, source_type, source_key
    """
    exact_bit_keys = [
        "embedded_bits",
        "num_embedded_bits",
        "payload_bits",
        "message_bits",
        "secret_bits",
        "capacity_bits",
        "actual_bits",
        "bits_embedded",
        "embedded_bit_count",
        "payload_bit_length",
        "message_bit_length",
        "secret_bit_length",
    ]
    bits, key = first_numeric_value(metadata, exact_bit_keys)
    if bits is not None and bits >= 0:
        return int(round(bits)), "metadata_exact", key or "unknown"

    bpn_keys = [
        "bpn",
        "BPN",
        "capacity_bpn",
        "payload_bpn",
        "bits_per_nucleotide",
    ]
    bpn, key = first_numeric_value(metadata, bpn_keys)
    if bpn is not None and bpn >= 0 and len(synthesized_dna) > 0:
        estimated_bits = int(round(bpn * len(synthesized_dna)))
        return estimated_bits, "metadata_bpn", key or "unknown"

    bitstring_keys = [
        "embedded_bitstream",
        "bitstream",
        "cipher_bits",
        "payload_bitstream",
        "message_bitstream",
        "secret_bitstream",
        "embedded_binary",
        "payload_binary",
    ]
    bit_length, key = first_bitstring_length(metadata, bitstring_keys)
    if bit_length is not None:
        return bit_length, "metadata_bitstring", key or "unknown"

    fallback_bits = len(secret_bytes) * 8
    return fallback_bits, "secret_file_fallback", "len(secret_bytes) * 8"


def safe_fake_ratio(metadata: Dict[str, Any]) -> Optional[float]:
    """安全读取 fake_ratio"""
    fake_ratio = metadata.get("fake_ratio")
    parsed = try_parse_number(fake_ratio)
    return parsed if parsed is not None else None


def batch_encode_dna(
    dna_txt_path: str,
    secret_path: str,
    output_path: str,
    secret_key: str,
    auto_optimize: bool = True,
) -> None:
    """批量执行 DNA 隐写并保存到文本文件，同时输出整体隐写容量 BPN"""
    print("=" * 80)
    print("开始批量生成隐写 DNA")
    print("=" * 80)
    print(f"输入 DNA 文件: {dna_txt_path}")
    print(f"秘密文件: {secret_path}")
    print(f"输出文件: {output_path}")
    print(f"自动优化 fake ratio: {'是' if auto_optimize else '否'}")

    secret_bytes = load_secret_bytes(secret_path)
    secret_bits = len(secret_bytes) * 8
    print(f"已读取 secret.txt 二进制内容: {len(secret_bytes)} 字节 ({secret_bits} bits)")

    dna_sequences = load_dna_sequences_from_txt(dna_txt_path)
    print(f"从 DNA.txt 共读取到 {len(dna_sequences)} 条有效 DNA 序列")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    stats_file = output_file.with_name(output_file.stem + "_stats.csv")

    success_count = 0
    skip_count = 0
    fail_count = 0

    total_reference_bases = 0
    total_stego_bases = 0
    total_embedded_bits = 0
    per_sequence_bpn_sum = 0.0
    exact_bits_count = 0
    fallback_bits_count = 0

    with open(output_file, "w", encoding="utf-8") as fout, open(
        stats_file, "w", encoding="utf-8-sig", newline=""
    ) as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "index",
                "status",
                "reference_length",
                "stego_length",
                "embedded_bits",
                "bpn",
                "fake_ratio",
                "bit_source_type",
                "bit_source_key",
            ]
        )

        for idx, reference_dna in enumerate(dna_sequences, start=1):
            try:
                if len(reference_dna) < MIN_VALID_BASES:
                    print(f"[跳过] 第 {idx} 条序列长度不足: {len(reference_dna)}")
                    skip_count += 1
                    writer.writerow([
                        idx,
                        "skipped",
                        len(reference_dna),
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ])
                    continue

                synthesized_dna, metadata = encode_message(
                    message_data=secret_bytes,
                    key=secret_key,
                    reference_dna=reference_dna,
                    auto_optimize=auto_optimize,
                )

                metadata = ensure_metadata_dict(metadata)
                fake_ratio = safe_fake_ratio(metadata)
                embedded_bits, bit_source_type, bit_source_key = infer_embedded_bits(
                    metadata=metadata,
                    secret_bytes=secret_bytes,
                    synthesized_dna=synthesized_dna,
                )

                stego_len = len(synthesized_dna)
                ref_len = len(reference_dna)
                bpn = (embedded_bits / stego_len) if stego_len > 0 else 0.0

                fout.write(synthesized_dna + "\n")
                success_count += 1

                total_reference_bases += ref_len
                total_stego_bases += stego_len
                total_embedded_bits += embedded_bits
                per_sequence_bpn_sum += bpn

                if bit_source_type == "secret_file_fallback":
                    fallback_bits_count += 1
                else:
                    exact_bits_count += 1

                writer.writerow(
                    [
                        idx,
                        "success",
                        ref_len,
                        stego_len,
                        embedded_bits,
                        f"{bpn:.6f}",
                        "" if fake_ratio is None else f"{fake_ratio:.6f}",
                        bit_source_type,
                        bit_source_key,
                    ]
                )

                message = (
                    f"[完成] 第 {idx} 条 | 原始长度={ref_len} | 隐写长度={stego_len} | "
                    f"嵌入bits={embedded_bits} | BPN={bpn:.6f}"
                )
                if fake_ratio is not None:
                    message += f" | fake_ratio={fake_ratio:.2f}"
                if bit_source_type == "secret_file_fallback":
                    message += " | bits来源=secret文件估算"
                else:
                    message += f" | bits来源={bit_source_key}"
                print(message)

            except Exception as e:
                fail_count += 1
                print(f"[失败] 第 {idx} 条处理出错: {e}")
                writer.writerow([idx, "failed", len(reference_dna), "", "", "", "", "", str(e)])

    overall_bpn = (total_embedded_bits / total_stego_bases) if total_stego_bases > 0 else 0.0
    average_bpn = (per_sequence_bpn_sum / success_count) if success_count > 0 else 0.0
    overall_bpn_ref = (total_embedded_bits / total_reference_bases) if total_reference_bases > 0 else 0.0

    print("\n" + "=" * 80)
    print("批量处理结束")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")
    print(f"累计参考序列总长度: {total_reference_bases}")
    print(f"累计隐写序列总长度: {total_stego_bases}")
    print(f"累计嵌入总比特数: {total_embedded_bits}")
    print(f"整体隐写容量 BPN(总bits / 总隐写长度): {overall_bpn:.6f}")
    print(f"平均单条 BPN: {average_bpn:.6f}")
    print(f"参考序列口径 BPN(总bits / 总原始长度): {overall_bpn_ref:.6f}")
    print(f"metadata精确/推导取值条数: {exact_bits_count}")
    print(f"secret文件回退估算条数: {fallback_bits_count}")
    print(f"隐写结果已保存到: {output_file.resolve()}")
    print(f"统计明细已保存到: {stats_file.resolve()}")
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
