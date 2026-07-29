#!/usr/bin/env python3
"""
DNA Data Hiding using Table Lookup Substitution Method (TLSM)
Based on: "Data Hiding in DNA Sequences Based on Table Lookup Substitution"
by Taur, Lin, Lee and Tao (2012)

直接运行此脚本即可执行隐写和提取操作。
修改下方的 CONFIG 配置参数来指定输入输出文件。
"""

import random
import json
import os
from typing import List, Tuple, Dict, Optional

# ============================================================================
# 配置参数 - 修改这里来指定你的文件路径和操作
# ============================================================================
CONFIG = {
    # 操作模式: 'hide' (隐藏数据), 'extract' (提取数据), 'demo' (运行演示)
    'mode': 'hide',

    # 隐写方法: 'tlsm' (基础2位), 'base3' (Base-3), 'etlsm' (扩展方法)
    'method': 'tlsm',

    # 随机种子 (用于生成位置集，确保可重复性)
    'seed': 42,

    # === 隐藏模式所需文件 ===
    'dna_file': r'.\4_Baselines\AAA36405.1.txt',  # 原始DNA序列文件
    'secret_file': r'.\4_Baselines\secret.txt',  # 秘密信息文件 (二进制或文本)
    'output_file': r'.\4_Baselines\output\TLSM.txt',  # 输出的隐写DNA文件

    # === 提取模式所需文件 ===
    'stego_file': r'TLSM_output\stego_output.txt',  # 隐写DNA文件
    'info_file': r'TLSM_output\stego_output_info.json',  # 隐写信息文件
    'extracted_file': 'extracted.txt',  # 提取的秘密信息输出文件
    'decode_as_text': True,  # 是否尝试将提取的二进制解码为文本
}


# ============================================================================


class TLSM_DNA_Steganography:
    """Table Lookup Substitution Method for DNA Data Hiding"""

    def __init__(self, rule_table: Optional[Dict] = None):
        """
        Initialize TLSM with optional custom rule table.

        Args:
            rule_table: Custom rule table dict, or None for default (Table 1 from paper)
        """
        if rule_table is None:
            # Default 2-bit rule table from the paper (Table 1)
            self.rule_table = {
                'A': {'00': 'C', '01': 'A', '10': 'T', '11': 'G'},
                'C': {'00': 'A', '01': 'C', '10': 'G', '11': 'T'},
                'G': {'00': 'T', '01': 'G', '10': 'A', '11': 'C'},
                'T': {'00': 'G', '01': 'T', '10': 'A', '11': 'C'}
            }
        else:
            self.rule_table = rule_table

        self.reverse_table = self._build_reverse_table()

    def _build_reverse_table(self) -> Dict:
        """Build reverse lookup table for extraction"""
        reverse = {}
        for orig_letter, mappings in self.rule_table.items():
            reverse[orig_letter] = {}
            for msg, sub_letter in mappings.items():
                reverse[orig_letter][sub_letter] = msg
        return reverse

    def phi(self, s: str, m1: str, m2: str) -> str:
        """Conversion function φ(s, m1, m2) - returns substituted letter"""
        msg_bits = m1 + m2
        return self.rule_table[s][msg_bits]

    def phi_reverse(self, s: str, s_prime: str) -> str:
        """Reversion function φ'(s, s') - returns 2-bit message"""
        return self.reverse_table[s][s_prime]

    def generate_location_set(self, n: int, p: int, seed: Optional[int] = None) -> List[int]:
        """Generate sorted ascending distinct integer set A"""
        if seed is not None:
            random.seed(seed)

        if p > n:
            raise ValueError(f"Cannot select {p} positions from sequence of length {n}")

        positions = sorted(random.sample(range(1, n + 1), p))
        return positions

    def hide_data(self, reference_dna: str, secret_binary: str,
                  location_set: Optional[List[int]] = None,
                  seed: Optional[int] = None) -> Tuple[str, List[int], int]:
        """
        Hide binary secret message in DNA sequence using TLSM (Algorithm 3).

        Args:
            reference_dna: Original DNA sequence S
            secret_binary: Binary message M
            location_set: Optional predefined location set A
            seed: Random seed for location generation

        Returns:
            Tuple of (faked_dna, location_set, num_bits)
        """
        reference_dna = self._clean_dna(reference_dna)

        if not all(c in 'ACGT' for c in reference_dna):
            raise ValueError("DNA sequence must contain only A, C, G, T")

        if not all(c in '01' for c in secret_binary):
            raise ValueError("Secret message must be binary (0s and 1s only)")

        # Ensure even length by padding
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

        # Algorithm 3: Embedding
        faked_dna = []
        location_idx = 0
        msg_idx = 0

        for i in range(1, n + 1):
            if location_idx < len(location_set) and i == location_set[location_idx]:
                orig_letter = reference_dna[i - 1]
                m1 = secret_binary[msg_idx]
                m2 = secret_binary[msg_idx + 1]
                new_letter = self.phi(orig_letter, m1, m2)
                faked_dna.append(new_letter)
                location_idx += 1
                msg_idx += 2
            else:
                faked_dna.append(random.choice(['A', 'C', 'G', 'T']))

        return ''.join(faked_dna), location_set, len(secret_binary)

    def extract_data(self, faked_dna: str, reference_dna: str,
                     location_set: List[int], num_bits: int) -> str:
        """
        Extract hidden Data from faked DNA sequence (Algorithm 4).
        """
        faked_dna = self._clean_dna(faked_dna)
        reference_dna = self._clean_dna(reference_dna)

        extracted_bits = []

        for pos in location_set:
            orig_letter = reference_dna[pos - 1]
            fake_letter = faked_dna[pos - 1]
            bits = self.phi_reverse(orig_letter, fake_letter)
            extracted_bits.append(bits)

        result = ''.join(extracted_bits)
        return result[:num_bits]

    def _clean_dna(self, dna: str) -> str:
        """Remove comments, whitespace, and convert to uppercase"""
        lines = dna.strip().split('\n')
        cleaned = []
        for line in lines:
            if line.strip().startswith('#'):
                continue
            cleaned.append(line.strip().upper())
        return ''.join(cleaned)


class Base3_TLSM(TLSM_DNA_Steganography):
    """Base-3 TLSM - encodes message in ternary for full table utilization"""

    def __init__(self):
        super().__init__()
        # Base-3 rule table from the paper (Table 6)
        self.rule_table = {
            'A': {'0': 'A', '1': 'T', '2': 'G', 'inf': 'C'},
            'C': {'0': 'C', '1': 'A', '2': 'T', 'inf': 'G'},
            'G': {'0': 'G', '1': 'C', '2': 'A', 'inf': 'T'},
            'T': {'0': 'T', '1': 'G', '2': 'C', 'inf': 'A'}
        }
        self.reverse_table = self._build_reverse_table()

    def binary_to_ternary(self, binary: str) -> str:
        """Convert binary string to base-3 representation"""
        if not binary:
            return '0'
        decimal = int(binary, 2)
        if decimal == 0:
            return '0'

        ternary = []
        while decimal > 0:
            ternary.append(str(decimal % 3))
            decimal //= 3
        return ''.join(reversed(ternary))

    def ternary_to_binary(self, ternary: str, original_bits: int) -> str:
        """Convert base-3 string back to binary"""
        decimal = int(ternary, 3) if ternary else 0
        binary = bin(decimal)[2:]
        return binary.zfill(original_bits)

    def hide_data(self, reference_dna: str, secret_binary: str,
                  location_set: Optional[List[int]] = None,
                  seed: Optional[int] = None) -> Tuple[str, List[int], int, int]:
        """Hide Data using Base-3 encoding"""
        reference_dna = self._clean_dna(reference_dna)

        if not all(c in '01' for c in secret_binary):
            raise ValueError("Secret message must be binary")

        original_binary_len = len(secret_binary)
        ternary_msg = self.binary_to_ternary(secret_binary)

        n = len(reference_dna)
        p = len(ternary_msg)

        if p > n:
            raise ValueError(f"Message too long. Max: {n} ternary digits")

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
                digit = ternary_msg[msg_idx]
                new_letter = self.rule_table[orig_letter][digit]
                faked_dna.append(new_letter)
                location_idx += 1
                msg_idx += 1
            else:
                orig_letter = reference_dna[i - 1]
                faked_dna.append(self.rule_table[orig_letter]['inf'])

        return ''.join(faked_dna), location_set, len(ternary_msg), original_binary_len

    def extract_data(self, faked_dna: str, reference_dna: str,
                     location_set: List[int], num_ternary: int,
                     original_binary_len: int) -> str:
        """Extract Data from Base-3 encoded sequence"""
        faked_dna = self._clean_dna(faked_dna)
        reference_dna = self._clean_dna(reference_dna)

        ternary_digits = []
        for pos in location_set[:num_ternary]:
            orig_letter = reference_dna[pos - 1]
            fake_letter = faked_dna[pos - 1]
            digit = self.reverse_table[orig_letter][fake_letter]
            ternary_digits.append(digit)

        ternary_str = ''.join(ternary_digits)
        return self.ternary_to_binary(ternary_str, original_binary_len)


class ETLSM(Base3_TLSM):
    """Extended TLSM - Uses neighboring letters for table selection"""

    def __init__(self, c: int = 1):
        super().__init__()
        self.c = c
        self.etlsm_tables = self._build_etlsm_tables()
        self.etlsm_reverse = self._build_etlsm_reverse()

    def _build_etlsm_tables(self) -> Dict:
        """Build ETLSM tables based on Table 7 from paper"""
        tables = {
            'A': {
                'A': {'0': 'G', '1': 'A', '2': 'T', 'inf': 'C'},
                'C': {'0': 'T', '1': 'G', '2': 'C', 'inf': 'A'},
                'G': {'0': 'T', '1': 'C', '2': 'A', 'inf': 'G'},
                'T': {'0': 'G', '1': 'T', '2': 'C', 'inf': 'A'}
            },
            'C': {
                'A': {'0': 'C', '1': 'A', '2': 'T', 'inf': 'G'},
                'C': {'0': 'A', '1': 'C', '2': 'T', 'inf': 'G'},
                'G': {'0': 'A', '1': 'T', '2': 'G', 'inf': 'C'},
                'T': {'0': 'G', '1': 'C', '2': 'A', 'inf': 'T'}
            },
            'G': {
                'A': {'0': 'C', '1': 'T', '2': 'G', 'inf': 'A'},
                'C': {'0': 'T', '1': 'A', '2': 'C', 'inf': 'G'},
                'G': {'0': 'A', '1': 'G', '2': 'C', 'inf': 'T'},
                'T': {'0': 'G', '1': 'T', '2': 'A', 'inf': 'C'}
            },
            'T': {
                'A': {'0': 'A', '1': 'C', '2': 'G', 'inf': 'T'},
                'C': {'0': 'C', '1': 'G', '2': 'T', 'inf': 'A'},
                'G': {'0': 'T', '1': 'A', '2': 'G', 'inf': 'C'},
                'T': {'0': 'C', '1': 'G', '2': 'A', 'inf': 'T'}
            }
        }
        return tables

    def _build_etlsm_reverse(self) -> Dict:
        """Build reverse lookup tables for ETLSM extraction"""
        reverse = {}
        for neighbor, letters in self.etlsm_tables.items():
            reverse[neighbor] = {}
            for orig_letter, mappings in letters.items():
                reverse[neighbor][orig_letter] = {}
                for msg, sub_letter in mappings.items():
                    reverse[neighbor][orig_letter][sub_letter] = msg
        return reverse

    def get_neighbor(self, dna: str, pos: int, n: int) -> str:
        """Get the neighboring letter using cyclic right neighbor"""
        neighbor_pos = (pos + 1) % n
        return dna[neighbor_pos]

    def hide_data(self, reference_dna: str, secret_binary: str,
                  location_set: Optional[List[int]] = None,
                  seed: Optional[int] = None) -> Tuple[str, List[int], int, int]:
        """Hide Data using ETLSM (Algorithm 7)"""
        reference_dna = self._clean_dna(reference_dna)

        if not all(c in '01' for c in secret_binary):
            raise ValueError("Secret message must be binary")

        original_binary_len = len(secret_binary)
        ternary_msg = self.binary_to_ternary(secret_binary)

        n = len(reference_dna)
        p = len(ternary_msg)

        if p > n:
            raise ValueError(f"Message too long")

        if location_set is None:
            location_set = self.generate_location_set(n, p, seed)

        if seed is not None:
            random.seed(seed + 1)

        faked_dna = list(reference_dna)

        for i in range(n):
            if (i + 1) not in location_set:
                neighbor = self.get_neighbor(reference_dna, i, n)
                orig_letter = reference_dna[i]
                faked_dna[i] = self.etlsm_tables[neighbor][orig_letter]['inf']

        for idx, pos in enumerate(location_set):
            i = pos - 1
            neighbor = self.get_neighbor(reference_dna, i, n)
            orig_letter = reference_dna[i]
            digit = ternary_msg[idx]
            faked_dna[i] = self.etlsm_tables[neighbor][orig_letter][digit]

        return ''.join(faked_dna), location_set, len(ternary_msg), original_binary_len

    def extract_data(self, faked_dna: str, reference_dna: str,
                     location_set: List[int], num_ternary: int,
                     original_binary_len: int) -> str:
        """Extract Data using ETLSM (Algorithm 8)"""
        faked_dna = self._clean_dna(faked_dna)
        reference_dna = self._clean_dna(reference_dna)
        n = len(reference_dna)

        ternary_digits = []
        for pos in location_set[:num_ternary]:
            i = pos - 1
            neighbor = self.get_neighbor(reference_dna, i, n)
            orig_letter = reference_dna[i]
            fake_letter = faked_dna[i]
            digit = self.etlsm_reverse[neighbor][orig_letter][fake_letter]
            ternary_digits.append(digit)

        ternary_str = ''.join(ternary_digits)
        return self.ternary_to_binary(ternary_str, original_binary_len)


# ============================================================================
# 辅助函数
# ============================================================================

def text_to_binary(text: str) -> str:
    """将文本转换为二进制字符串"""
    return ''.join(format(ord(c), '08b') for c in text)


def binary_to_text(binary: str) -> str:
    """将二进制字符串转换为文本"""
    chars = []
    for i in range(0, len(binary) - len(binary) % 8, 8):
        byte = binary[i:i + 8]
        chars.append(chr(int(byte, 2)))
    return ''.join(chars)


def read_dna_file(filepath: str) -> str:
    """从文件读取DNA序列"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def read_secret_file(filepath: str) -> str:
    """读取秘密信息文件 (支持二进制和文本格式)"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # 检查是否已经是二进制
    clean_content = content.replace('\n', '').replace(' ', '')
    if all(c in '01' for c in clean_content):
        return clean_content

    # 将文本转换为二进制
    return text_to_binary(content)


def generate_random_dna(length: int) -> str:
    """生成随机DNA序列"""
    return ''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(length))


# ============================================================================
# 主要操作函数
# ============================================================================

def hide_data_in_dna(dna_file: str, secret_file: str, output_file: str,
                     method: str = 'tlsm', seed: int = 42):
    """
    将秘密信息隐藏到DNA序列中

    Args:
        dna_file: 原始DNA序列文件路径
        secret_file: 秘密信息文件路径
        output_file: 输出文件路径
        method: 隐写方法 ('tlsm', 'base3', 'etlsm')
        seed: 随机种子
    """
    print("=" * 60)
    print("DNA数据隐藏 - TLSM方法")
    print("=" * 60)

    # 读取输入文件
    reference_dna = read_dna_file(dna_file)
    secret_binary = read_secret_file(secret_file)

    print(f"\n输入DNA文件: {dna_file}")
    print(f"秘密文件: {secret_file}")
    print(f"秘密信息二进制长度: {len(secret_binary)} bits")

    # 选择方法并执行隐藏
    if method == 'tlsm':
        stego = TLSM_DNA_Steganography()
        faked_dna, locations, num_bits = stego.hide_data(
            reference_dna, secret_binary, seed=seed
        )
        extra_info = {'num_bits': num_bits}
        method_name = 'TLSM (2-bit)'

    elif method == 'base3':
        stego = Base3_TLSM()
        faked_dna, locations, num_ternary, orig_len = stego.hide_data(
            reference_dna, secret_binary, seed=seed
        )
        extra_info = {'num_ternary': num_ternary, 'original_bits': orig_len}
        method_name = 'Base-3 TLSM'

    elif method == 'etlsm':
        stego = ETLSM()
        faked_dna, locations, num_ternary, orig_len = stego.hide_data(
            reference_dna, secret_binary, seed=seed
        )
        extra_info = {'num_ternary': num_ternary, 'original_bits': orig_len}
        method_name = 'ETLSM (Extended)'
    else:
        raise ValueError(f"未知方法: {method}")

    # 计算统计信息
    clean_ref = stego._clean_dna(reference_dna)
    modifications = sum(1 for a, b in zip(clean_ref, faked_dna) if a != b)
    mod_rate = (modifications / len(faked_dna)) * 100
    bpn = len(secret_binary) / len(faked_dna)

    # 写入输出文件
    output_base = os.path.splitext(output_file)[0]

    # 写入隐写DNA
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# DNA隐写输出 (TLSM方法)\n")
        f.write(f"# 方法: {method_name}\n")
        f.write(f"# 隐藏位数: {len(secret_binary)}\n")
        f.write(f"# 位置数: {len(locations)}\n")
        f.write(f"# BPN: {bpn:.4f}\n")
        f.write(f"# 修改率: {mod_rate:.2f}%\n\n")
        f.write(faked_dna)

    # 写入位置集
    loc_file = f"{output_base}_locations.txt"
    with open(loc_file, 'w', encoding='utf-8') as f:
        f.write(f"# 位置集 (1-索引)\n")
        f.write(f"# 总位置数: {len(locations)}\n\n")
        f.write(','.join(map(str, locations)))

    # 写入提取信息 (用于后续提取)
    info_file = f"{output_base}_info.json"
    info = {
        'method': method,
        'seed': seed,
        'locations': locations,
        **extra_info
    }
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 40}")
    print("隐藏结果")
    print(f"{'=' * 40}")
    print(f"方法: {method_name}")
    print(f"DNA长度: {len(faked_dna)}")
    print(f"隐藏位数: {len(secret_binary)}")
    print(f"使用位置数: {len(locations)}")
    print(f"BPN (每核苷酸位数): {bpn:.4f}")
    print(f"修改率: {mod_rate:.2f}%")
    print(f"\n输出文件:")
    print(f"  - 隐写DNA: {output_file}")
    print(f"  - 位置集: {loc_file}")
    print(f"  - 信息文件: {info_file}")
    print(f"{'=' * 40}")

    return faked_dna, locations, info


def extract_data_from_dna(dna_file: str, stego_file: str, info_file: str,
                          output_file: str, decode_as_text: bool = True):
    """
    从隐写DNA序列中提取秘密信息

    Args:
        dna_file: 原始DNA序列文件路径
        stego_file: 隐写DNA文件路径
        info_file: 信息JSON文件路径
        output_file: 输出文件路径
        decode_as_text: 是否尝试解码为文本
    """
    print("=" * 60)
    print("DNA数据提取 - TLSM方法")
    print("=" * 60)

    # 读取输入文件
    reference_dna = read_dna_file(dna_file)
    faked_dna = read_dna_file(stego_file)

    with open(info_file, 'r', encoding='utf-8') as f:
        info = json.load(f)

    method = info['method']
    locations = info['locations']

    print(f"\n原始DNA文件: {dna_file}")
    print(f"隐写DNA文件: {stego_file}")
    print(f"方法: {method}")

    # 根据方法提取数据
    if method == 'tlsm':
        stego = TLSM_DNA_Steganography()
        extracted = stego.extract_data(
            faked_dna, reference_dna, locations, info['num_bits']
        )
    elif method == 'base3':
        stego = Base3_TLSM()
        extracted = stego.extract_data(
            faked_dna, reference_dna, locations,
            info['num_ternary'], info['original_bits']
        )
    elif method == 'etlsm':
        stego = ETLSM()
        extracted = stego.extract_data(
            faked_dna, reference_dna, locations,
            info['num_ternary'], info['original_bits']
        )
    else:
        raise ValueError(f"未知方法: {method}")

    # 写入提取的二进制
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(extracted)

    print(f"\n{'=' * 40}")
    print("提取结果")
    print(f"{'=' * 40}")
    print(f"提取位数: {len(extracted)}")
    print(f"输出文件: {output_file}")

    # 尝试解码为文本
    if decode_as_text:
        try:
            text = binary_to_text(extracted)
            text_file = os.path.splitext(output_file)[0] + '_text.txt'
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"\n解码后的文本已保存到: {text_file}")
            print(f"内容: {text[:100]}{'...' if len(text) > 100 else ''}")
        except Exception as e:
            print(f"\n无法解码为文本: {e}")

    print(f"{'=' * 40}")

    return extracted


def run_demo():
    """运行论文示例演示"""
    print("=" * 60)
    print("TLSM演示 - 论文示例 (Example 1)")
    print("=" * 60)

    # 论文中的示例 (Section 3.2)
    reference_dna = "ACGGAATTGCTTCAG"
    secret_binary = "01110100101110"
    location_set = [2, 3, 5, 10, 12, 13, 15]
    expected_output = "TCCCACATCAAATAA"

    print(f"\n原始DNA序列: {reference_dna}")
    print(f"秘密二进制: {secret_binary}")
    print(f"位置集: {location_set}")
    print(f"论文预期输出: {expected_output}")

    # 测试TLSM
    tlsm = TLSM_DNA_Steganography()
    faked_dna, _, num_bits = tlsm.hide_data(
        reference_dna, secret_binary, location_set
    )

    print(f"\n生成的DNA: {faked_dna}")

    # 逐位置比较
    print("\n位置对比 (仅隐藏位置):")
    for pos in location_set:
        orig = reference_dna[pos - 1]
        fake = faked_dna[pos - 1]
        exp = expected_output[pos - 1]
        match = "✓" if fake == exp else "✗"
        print(f"  位置 {pos}: {orig} -> {fake} (预期: {exp}) {match}")

    # 提取并验证
    extracted = tlsm.extract_data(faked_dna, reference_dna, location_set, num_bits)
    print(f"\n提取结果: {extracted}")
    print(f"原始信息: {secret_binary}")
    print(f"匹配: {'✓ 成功' if extracted == secret_binary else '✗ 失败'}")

    # 额外测试: 使用文本消息
    print("\n" + "=" * 60)
    print("额外测试: 文本消息隐藏")
    print("=" * 60)

    # 生成随机DNA
    random.seed(123)
    long_dna = generate_random_dna(500)
    secret_text = "Hello TLSM!"
    secret_bin = text_to_binary(secret_text)

    print(f"\nDNA长度: {len(long_dna)}")
    print(f"秘密消息: '{secret_text}'")
    print(f"二进制长度: {len(secret_bin)} bits")

    # 隐藏
    faked_long, locs_long, bits_long = tlsm.hide_data(long_dna, secret_bin, seed=999)

    # 提取
    extracted_bin = tlsm.extract_data(faked_long, long_dna, locs_long, bits_long)
    extracted_text = binary_to_text(extracted_bin)

    print(f"提取的消息: '{extracted_text}'")
    print(f"匹配: {'✓ 成功' if extracted_text == secret_text else '✗ 失败'}")

    # 测试所有三种方法
    print("\n" + "=" * 60)
    print("三种方法对比测试")
    print("=" * 60)

    test_secret = "Test123"
    test_bin = text_to_binary(test_secret)
    test_dna = generate_random_dna(300)

    methods = [
        ('tlsm', TLSM_DNA_Steganography()),
        ('base3', Base3_TLSM()),
        ('etlsm', ETLSM())
    ]

    for name, method_obj in methods:
        if name == 'tlsm':
            faked, locs, num = method_obj.hide_data(test_dna, test_bin, seed=42)
            extracted = method_obj.extract_data(faked, test_dna, locs, num)
        else:
            faked, locs, num_t, orig = method_obj.hide_data(test_dna, test_bin, seed=42)
            extracted = method_obj.extract_data(faked, test_dna, locs, num_t, orig)

        result_text = binary_to_text(extracted)
        status = "✓" if result_text == test_secret else "✗"
        print(f"  {name.upper():8} -> '{result_text}' {status}")


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    mode = CONFIG['mode']

    if mode == 'demo':
        # 运行演示
        run_demo()

    elif mode == 'hide':
        # 隐藏数据
        hide_data_in_dna(
            dna_file=CONFIG['dna_file'],
            secret_file=CONFIG['secret_file'],
            output_file=CONFIG['output_file'],
            method=CONFIG['method'],
            seed=CONFIG['seed']
        )

    elif mode == 'extract':
        # 提取数据
        extract_data_from_dna(
            dna_file=CONFIG['dna_file'],
            stego_file=CONFIG['stego_file'],
            info_file=CONFIG['info_file'],
            output_file=CONFIG['extracted_file'],
            decode_as_text=CONFIG['decode_as_text']
        )

    else:
        print(f"未知模式: {mode}")
        print("可用模式: 'demo', 'hide', 'extract'")