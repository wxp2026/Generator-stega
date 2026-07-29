#!/usr/bin/env python3
import argparse
import os
import random
from typing import List, Tuple, Dict, Optional


class TLSM_DNA_Steganography:
    def __init__(self, rule_table: Optional[Dict] = None):
        if rule_table is None:
            self.rule_table = {
                'A': {'00': 'C', '01': 'A', '10': 'T', '11': 'G'},
                'C': {'00': 'A', '01': 'C', '10': 'G', '11': 'T'},
                'G': {'00': 'T', '01': 'G', '10': 'A', '11': 'C'},
                'T': {'00': 'G', '01': 'T', '10': 'A', '11': 'C'}
            }
        else:
            self.rule_table = rule_table

    def phi(self, s: str, m1: str, m2: str) -> str:
        return self.rule_table[s][m1 + m2]

    def generate_location_set(self, n: int, p: int, seed: Optional[int] = None) -> List[int]:
        if seed is not None:
            random.seed(seed)
        if p > n:
            raise ValueError(f"Cannot select {p} positions from sequence of length {n}")
        return sorted(random.sample(range(1, n + 1), p))

    def _clean_dna(self, dna: str) -> str:
        lines = dna.strip().split('\n')
        cleaned = []
        for line in lines:
            if line.strip().startswith('#'):
                continue
            cleaned.append(line.strip().upper())
        return ''.join(cleaned)

    def hide_data(self, reference_dna: str, secret_binary: str, location_set: Optional[List[int]] = None,
                  seed: Optional[int] = None) -> Tuple[str, List[int], int]:
        reference_dna = self._clean_dna(reference_dna)
        if not all(c in 'ACGT' for c in reference_dna):
            raise ValueError("DNA sequence must contain only A, C, G, T")
        if not all(c in '01' for c in secret_binary):
            raise ValueError("Secret message must be binary (0/1 only)")
        if len(secret_binary) % 2 != 0:
            secret_binary += '0'

        n = len(reference_dna)
        p = len(secret_binary) // 2
        if p > n:
            raise ValueError(f"Message too long. Max capacity: {n * 2} bits, got {len(secret_binary)} bits")

        if location_set is None:
            location_set = self.generate_location_set(n, p, seed)

        if seed is not None:
            random.seed(seed + 1)

        faked_dna = []
        location_idx = 0
        msg_idx = 0
        for i in range(1, n + 1):
            if location_idx < len(location_set) and i == location_set[location_idx]:
                orig_letter = reference_dna[i - 1]
                m1 = secret_binary[msg_idx]
                m2 = secret_binary[msg_idx + 1]
                faked_dna.append(self.phi(orig_letter, m1, m2))
                location_idx += 1
                msg_idx += 2
            else:
                faked_dna.append(random.choice(['A', 'C', 'G', 'T']))

        return ''.join(faked_dna), location_set, len(secret_binary)


def text_to_binary(text: str) -> str:
    return ''.join(format(ord(c), '08b') for c in text)



def read_secret_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    clean_content = content.replace('\n', '').replace(' ', '')
    if clean_content and all(c in '01' for c in clean_content):
        return clean_content
    return text_to_binary(content)



def sanitize_dna(seq: str) -> Optional[str]:
    if seq is None:
        return None
    seq = str(seq).strip().upper()
    if not seq:
        return None
    seq = seq.replace('U', 'T')
    seq = ''.join(ch for ch in seq if not ch.isspace())
    if any(ch not in 'ATGC' for ch in seq):
        return None
    return seq



def hide_dna_txt_to_txt(dna_file: str, secret_file: str, output_file: str, seed: int = 42) -> None:
    secret_binary = read_secret_file(secret_file)
    stego = TLSM_DNA_Steganography()

    total_rows = success_rows = skipped_invalid = skipped_too_short = 0
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    with open(dna_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, line in enumerate(f_in, start=1):
            total_rows += 1
            clean_seq = sanitize_dna(line)
            if clean_seq is None:
                skipped_invalid += 1
                print(f"[跳过] 第 {idx} 行包含非 ATGC 字符或为空")
                continue
            try:
                stego_dna, _, _ = stego.hide_data(clean_seq, secret_binary, seed=seed + idx)
                f_out.write(stego_dna + '\n')
                success_rows += 1
            except ValueError as e:
                skipped_too_short += 1
                print(f"[跳过] 第 {idx} 行容量不足或数据异常: {e}")

    print("=" * 60)
    print("处理完成")
    print(f"总序列数: {total_rows}")
    print(f"成功隐写: {success_rows}")
    print(f"跳过(非法字符/空): {skipped_invalid}")
    print(f"跳过(容量不足): {skipped_too_short}")
    print(f"输入文件: {dna_file}")
    print(f"输出文件: {output_file}")
    print("=" * 60)



def main():
    parser = argparse.ArgumentParser(description="读取 DNA.txt 每一行 DNA 序列进行 TLSM 隐写，并写入目标.txt")
    parser.add_argument('--dna-file', required=True,
                        default=r'.\5_PCA _ SWD\raw_clean.txt',
                        help='DNA.txt 文件路径（每行一条 DNA 序列）')
    parser.add_argument('--secret-file', required=True,
                        default=r'.\4_Baselines\secret.txt',
                        help='秘密信息文件路径（二进制或文本）')
    parser.add_argument('--output',
                        default=r'.\5_PCA _ SWD\TLSM.txt',
                        help='输出目标.txt 路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    hide_dna_txt_to_txt(args.dna_file, args.secret_file, args.output, args.seed)


if __name__ == '__main__':
    main()
