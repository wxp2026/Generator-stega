#!/usr/bin/env python3
"""
BioCode: DNA Steganography Implementation
Based on: Haughton and Balado, "BioCode: Two biologically compatible algorithms
for embedding Data in non-coding and coding regions of DNA", BMC Bioinformatics 2013

简化版：只需传入DNA.txt和secret.txt文件路径即可运行
"""

import os
from typing import List, Tuple, Dict
from collections import Counter
import math

# ============================================================================
# 配置区域 - 在这里直接修改输入输出文件路径
# ============================================================================

DNA_FILE = r".\4_Baselines\AAA36405.1.txt"  # 原始DNA序列文件
SECRET_FILE = r".\4_Baselines\secret.txt"  # 秘密信息文件（二进制字符串）
OUTPUT_FILE = r".\4_Baselines\output\NSCRS\NSCRS_AAA36405.1.txt"  # 编码后的DNA序列输出文件
MODE = 'ncdna'  # 'ncdna', 'pcdna', 或 'auto'

# ============================================================================
# 常量定义
# ============================================================================

BASES = ['A', 'C', 'T', 'G']
START_CODONS = {'ATG', 'CTG', 'TTG'}
COMPLEMENT_START = {'CAT', 'CAG', 'CAA'}
SPECIAL_DINUCLEOTIDES = {'AT', 'CT', 'TT', 'CA'}

GENETIC_CODE = {
    'TTT': 'Phe', 'TTC': 'Phe',
    'TTA': 'Leu', 'TTG': 'Leu', 'CTT': 'Leu', 'CTC': 'Leu', 'CTA': 'Leu', 'CTG': 'Leu',
    'ATT': 'Ile', 'ATC': 'Ile', 'ATA': 'Ile',
    'ATG': 'Met',
    'GTT': 'Val', 'GTC': 'Val', 'GTA': 'Val', 'GTG': 'Val',
    'TCT': 'Ser', 'TCC': 'Ser', 'TCA': 'Ser', 'TCG': 'Ser', 'AGT': 'Ser', 'AGC': 'Ser',
    'CCT': 'Pro', 'CCC': 'Pro', 'CCA': 'Pro', 'CCG': 'Pro',
    'ACT': 'Thr', 'ACC': 'Thr', 'ACA': 'Thr', 'ACG': 'Thr',
    'GCT': 'Ala', 'GCC': 'Ala', 'GCA': 'Ala', 'GCG': 'Ala',
    'TAT': 'Tyr', 'TAC': 'Tyr',
    'TAA': 'Stp', 'TAG': 'Stp', 'TGA': 'Stp',
    'CAT': 'His', 'CAC': 'His',
    'CAA': 'Gln', 'CAG': 'Gln',
    'AAT': 'Asn', 'AAC': 'Asn',
    'AAA': 'Lys', 'AAG': 'Lys',
    'GAT': 'Asp', 'GAC': 'Asp',
    'GAA': 'Glu', 'GAG': 'Glu',
    'TGT': 'Cys', 'TGC': 'Cys',
    'TGG': 'Trp',
    'CGT': 'Arg', 'CGC': 'Arg', 'CGA': 'Arg', 'CGG': 'Arg', 'AGA': 'Arg', 'AGG': 'Arg',
    'GGT': 'Gly', 'GGC': 'Gly', 'GGA': 'Gly', 'GGG': 'Gly',
}

AMINO_TO_CODONS = {}
for codon, amino in GENETIC_CODE.items():
    if amino not in AMINO_TO_CODONS:
        AMINO_TO_CODONS[amino] = []
    AMINO_TO_CODONS[amino].append(codon)
for amino in AMINO_TO_CODONS:
    AMINO_TO_CODONS[amino].sort()


# ============================================================================
# 核心算法函数
# ============================================================================

def graduated_mapping(symbols: List[str]) -> Dict[str, str]:
    """创建从符号到二进制字符串的分级映射"""
    mu = len(symbols)
    if mu == 0:
        return {}
    if mu == 1:
        return {'': symbols[0]}

    l = int(math.floor(math.log2(mu)))
    num_short = 2 ** (l + 1) - mu
    mapping = {}
    symbol_idx = 0

    for i in range(num_short):
        binary_str = format(i, f'0{l}b')
        mapping[binary_str] = symbols[symbol_idx]
        symbol_idx += 1

    for i in range(mu - num_short):
        code_val = num_short * 2 + i
        binary_str = format(code_val, f'0{l + 1}b')
        mapping[binary_str] = symbols[symbol_idx]
        symbol_idx += 1

    return mapping


def reverse_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    return {v: k for k, v in mapping.items()}


# ============================================================================
# BioCode ncDNA 类
# ============================================================================

class BioCodeNcDNA:
    """BioCode ncDNA: 在非编码DNA区域嵌入数据"""

    def __init__(self):
        self.allowed_bases = {
            'AT': ['A', 'T', 'C'],
            'CT': ['A', 'T', 'C'],
            'TT': ['A', 'T', 'C'],
            'CA': ['C'],
        }
        self.mappings = {}
        for dinuc, bases in self.allowed_bases.items():
            self.mappings[dinuc] = graduated_mapping(bases)
        self.default_mapping = graduated_mapping(BASES)

    def _get_context_mapping(self, dinuc: str) -> Dict[str, str]:
        if dinuc in SPECIAL_DINUCLEOTIDES:
            return self.mappings[dinuc]
        return self.default_mapping

    def encode(self, message_bits: str) -> str:
        dna_sequence = []
        bit_idx = 0

        while bit_idx < len(message_bits):
            if len(dna_sequence) >= 2:
                dinuc = dna_sequence[-2] + dna_sequence[-1]
            else:
                dinuc = ''

            mapping = self._get_context_mapping(dinuc)
            matched = False

            # 尝试匹配最长的二进制串
            for length in sorted(set(len(k) for k in mapping.keys() if k), reverse=True):
                if bit_idx + length <= len(message_bits):
                    bits = message_bits[bit_idx:bit_idx + length]
                    if bits in mapping:
                        dna_sequence.append(mapping[bits])
                        bit_idx += length
                        matched = True
                        break

            # 如果没有匹配，使用默认或任意推进
            if not matched:
                if '' in mapping:
                    dna_sequence.append(mapping[''])
                elif '0' in mapping:
                    dna_sequence.append(mapping['0'])
                elif '1' in mapping:
                    dna_sequence.append(mapping['1'])
                else:
                    dna_sequence.append(BASES[0])
                # **强制推进 bit_idx 防止死循环**
                bit_idx += 1

        return ''.join(dna_sequence)

    def decode(self, dna_sequence: str) -> str:
        message_bits = []
        decoded_so_far = []

        for base in dna_sequence:
            if len(decoded_so_far) >= 2:
                dinuc = decoded_so_far[-2] + decoded_so_far[-1]
            else:
                dinuc = ''

            mapping = self._get_context_mapping(dinuc)
            rev_mapping = reverse_mapping(mapping)

            if base in rev_mapping:
                bits = rev_mapping[base]
                if bits:
                    message_bits.append(bits)

            decoded_so_far.append(base)

        return ''.join(message_bits)


# ============================================================================
# BioCode pcDNA 类
# ============================================================================

class BioCodePcDNA:
    """BioCode pcDNA: 在蛋白质编码DNA区域嵌入数据"""

    def _get_amino_acid(self, codon: str) -> str:
        return GENETIC_CODE.get(codon.upper(), None)

    def _parse_codons(self, dna_sequence: str) -> List[str]:
        dna_sequence = dna_sequence.upper().replace(' ', '').replace('\n', '')
        codons = []
        for i in range(0, len(dna_sequence) - 2, 3):
            codons.append(dna_sequence[i:i + 3])
        return codons

    def encode(self, host_dna: str, message_bits: str) -> Tuple[str, dict]:
        host_codons = self._parse_codons(host_dna)
        amino_acids = [self._get_amino_acid(c) for c in host_codons]
        codon_counts = Counter(host_codons)

        available_codons = {}
        for codon, count in codon_counts.items():
            amino = self._get_amino_acid(codon)
            if amino not in available_codons:
                available_codons[amino] = {}
            available_codons[amino][codon] = count

        encoded_codons = []
        bit_idx = 0
        stats = {'bits_encoded': 0, 'codons_used': 0}

        for i, amino in enumerate(amino_acids):
            if amino is None:
                encoded_codons.append(host_codons[i])
                continue

            avail = {c: cnt for c, cnt in available_codons.get(amino, {}).items() if cnt > 0}

            if not avail:
                encoded_codons.append(host_codons[i])
                continue

            codon_list = sorted(avail.keys())
            mapping = graduated_mapping(codon_list)

            if len(codon_list) == 1:
                selected_codon = codon_list[0]
            else:
                selected_codon = None
                remaining_bits = message_bits[bit_idx:]

                for length in sorted(set(len(k) for k in mapping.keys()), reverse=True):
                    if len(remaining_bits) >= length:
                        bits = remaining_bits[:length]
                        if bits in mapping:
                            selected_codon = mapping[bits]
                            bit_idx += length
                            stats['bits_encoded'] += length
                            break

                if selected_codon is None:
                    selected_codon = codon_list[0]

            available_codons[amino][selected_codon] -= 1
            encoded_codons.append(selected_codon)
            stats['codons_used'] += 1

        encoded_dna = ''.join(encoded_codons)
        stats['total_bits'] = len(message_bits)
        stats['embedding_rate'] = stats['bits_encoded'] / len(encoded_codons) if encoded_codons else 0

        return encoded_dna, stats

    def decode(self, host_dna: str, encoded_dna: str) -> str:
        host_codons = self._parse_codons(host_dna)
        encoded_codons = self._parse_codons(encoded_dna)
        amino_acids = [self._get_amino_acid(c) for c in host_codons]
        codon_counts = Counter(host_codons)

        available_codons = {}
        for codon, count in codon_counts.items():
            amino = self._get_amino_acid(codon)
            if amino not in available_codons:
                available_codons[amino] = {}
            available_codons[amino][codon] = count

        message_bits = []

        for i, amino in enumerate(amino_acids):
            if amino is None or i >= len(encoded_codons):
                continue

            avail = {c: cnt for c, cnt in available_codons.get(amino, {}).items() if cnt > 0}

            if len(avail) <= 1:
                if avail:
                    codon = list(avail.keys())[0]
                    available_codons[amino][codon] -= 1
                continue

            codon_list = sorted(avail.keys())
            mapping = graduated_mapping(codon_list)
            rev_mapping = reverse_mapping(mapping)

            encoded_codon = encoded_codons[i]
            if encoded_codon in rev_mapping:
                message_bits.append(rev_mapping[encoded_codon])

            if encoded_codon in available_codons.get(amino, {}):
                available_codons[amino][encoded_codon] -= 1

        return ''.join(message_bits)


# ============================================================================
# 文件读取函数
# ============================================================================

def read_dna_file(filepath: str) -> str:
    """读取DNA序列文件"""
    with open(filepath, 'r') as f:
        content = f.read()
    return ''.join(c for c in content.upper() if c in 'ACGT')


def read_secret_file(filepath: str) -> str:
    """读取秘密文件（二进制字符串或普通文本）"""
    with open(filepath, 'r') as f:
        content = f.read().strip()

    # 检查是否已经是二进制字符串
    clean = content.replace('\n', '').replace(' ', '')
    if all(c in '01' for c in clean):
        return clean
    else:
        # 转换为二进制
        return ''.join(format(ord(c), '08b') for c in content)


# ============================================================================
# 主编码函数
# ============================================================================

def encode_dna(dna_file: str, secret_file: str, output_file: str = None, mode: str = 'auto') -> str:
    """
    DNA隐写编码主函数

    Args:
        dna_file: 原始DNA序列文件路径
        secret_file: 秘密信息文件路径
        output_file: 输出文件路径（可选，默认为 encoded_dna.txt）
        mode: 'ncdna', 'pcdna', 或 'auto'

    Returns:
        输出文件路径
    """
    if output_file is None:
        # 在DNA文件同目录生成输出文件
        dir_path = os.path.dirname(dna_file)
        output_file = os.path.join(dir_path, "encoded_dna.txt") if dir_path else "encoded_dna.txt"

    # 读取文件
    dna_sequence = read_dna_file(dna_file)
    secret_bits = read_secret_file(secret_file)

    print(f"DNA序列长度: {len(dna_sequence)} 碱基")
    print(f"秘密信息长度: {len(secret_bits)} 位")

    # 确定模式
    if mode == 'auto':
        mode = 'pcdna' if len(dna_sequence) % 3 == 0 and len(dna_sequence) >= 30 else 'ncdna'

    print(f"使用模式: {mode.upper()}")

    # 编码
    if mode == 'ncdna':
        encoder = BioCodeNcDNA()
        encoded_dna = encoder.encode(secret_bits)
        print(f"编码后DNA长度: {len(encoded_dna)} 碱基")
        print(f"嵌入率: {len(secret_bits) / len(encoded_dna):.4f} bits/base")
    else:
        encoder = BioCodePcDNA()
        encoded_dna, stats = encoder.encode(dna_sequence, secret_bits)
        print(f"编码位数: {stats['bits_encoded']} / {len(secret_bits)}")
        print(f"嵌入率: {stats['embedding_rate']:.4f} bits/codon")

    # 保存结果
    with open(output_file, 'w') as f:
        f.write(encoded_dna)

    print(f"编码后的DNA已保存到: {output_file}")
    return output_file


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    # 直接使用配置区域的文件路径
    if os.path.exists(DNA_FILE) and os.path.exists(SECRET_FILE):
        encode_dna(DNA_FILE, SECRET_FILE, OUTPUT_FILE, MODE)
    else:
        print(f"错误: 未找到输入文件")
        print(f"  DNA文件: {DNA_FILE}")
        print(f"  秘密文件: {SECRET_FILE}")
        print(f"\n请修改代码顶部的 DNA_FILE 和 SECRET_FILE 变量为正确的文件路径")