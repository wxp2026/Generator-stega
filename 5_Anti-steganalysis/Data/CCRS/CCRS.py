#!/usr/bin/env python3
"""
DNA-Courier Steganography Implementation (改进版)
特性:
1. 自动生成最优Mapping Table
2. 自动优化Fake codon数量
3. 比较不同方案的隐写效果
"""

import random
import hashlib
from typing import Tuple, List, Dict
from collections import Counter

# 尝试导入加密库
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import padding

    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    print("警告: cryptography库未安装,将使用简单XOR加密")

# 起始和终止密码子(需要避免)
START_STOP_CODONS = {'ATG', 'TAA', 'TAG', 'TGA'}

# 所有64个可能的密码子
ALL_CODONS = [a + b + c for a in 'ACGT' for b in 'ACGT' for c in 'ACGT']


def pad_key(key: str) -> bytes:
    """将密钥填充或哈希为16字节"""
    key_bytes = key.encode('utf-8')
    if len(key_bytes) < 16:
        key_bytes = key_bytes + b'\x00' * (16 - len(key_bytes))
    elif len(key_bytes) > 16:
        key_bytes = hashlib.md5(key_bytes).digest()
    return key_bytes[:16]


def aes_encrypt(plaintext: bytes, key: str) -> bytes:
    """使用AES-128加密"""
    if HAS_CRYPTO:
        key_bytes = pad_key(key)
        iv = b'\x00' * 16
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext) + padder.finalize()
        cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        return encryptor.update(padded_data) + encryptor.finalize()
    else:
        key_bytes = pad_key(key)
        pad_len = 16 - (len(plaintext) % 16)
        plaintext_bytes = plaintext + bytes([pad_len] * pad_len)
        encrypted = bytearray()
        for i, b in enumerate(plaintext_bytes):
            encrypted.append(b ^ key_bytes[i % 16])
        return bytes(encrypted)


def aes_decrypt(ciphertext: bytes, key: str) -> bytes:
    """使用AES-128解密"""
    if HAS_CRYPTO:
        key_bytes = pad_key(key)
        iv = b'\x00' * 16
        cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()
    else:
        key_bytes = pad_key(key)
        decrypted = bytearray()
        for i, b in enumerate(ciphertext):
            decrypted.append(b ^ key_bytes[i % 16])
        pad_len = decrypted[-1]
        return bytes(decrypted[:-pad_len])


def bytes_to_binary(data: bytes) -> str:
    """字节转二进制字符串"""
    return ''.join(format(b, '08b') for b in data)


def binary_to_bytes(binary: str) -> bytes:
    """二进制字符串转字节"""
    while len(binary) % 8 != 0:
        binary = '0' + binary
    result = bytearray()
    for i in range(0, len(binary), 8):
        result.append(int(binary[i:i + 8], 2))
    return bytes(result)


def analyze_codon_frequency(dna_sequence: str) -> Dict[str, float]:
    """分析DNA序列中的密码子频率"""
    codons = [dna_sequence[i:i + 3] for i in range(0, len(dna_sequence) - 2, 3)]
    total = len(codons)
    if total == 0:
        return {}
    counter = Counter(codons)
    return {codon: count / total for codon, count in counter.items()}


def analyze_5bit_frequency(ciphertext_list: List[bytes]) -> Dict[str, float]:
    """分析多个密文的5位值频率分布"""
    all_bits = ''
    for ciphertext in ciphertext_list:
        all_bits += bytes_to_binary(ciphertext)

    # 填充到5的倍数
    padding = (5 - len(all_bits) % 5) % 5
    all_bits += '0' * padding

    # 统计5位值频率
    bit5_counter = Counter()
    for i in range(0, len(all_bits), 5):
        bit5_counter[all_bits[i:i + 5]] += 1

    total = sum(bit5_counter.values())
    return {k: v / total for k, v in bit5_counter.items()}


def generate_mapping_table(reference_dna: str, num_samples: int = 100) -> Dict[str, str]:
    """
    自动生成最优Mapping Table

    策略:
    1. 分析参考DNA的codon频率
    2. 生成多个随机密文样本,分析5位值频率
    3. 选择频率最高的32个codons(排除start/stop)
    4. 将它们与5位值按频率匹配
    """
    print("\n🔬 开始生成最优Mapping Table...")

    # 1. 分析参考DNA的codon频率
    codon_freq = analyze_codon_frequency(reference_dna)
    print(f"   ✓ 分析了 {len(codon_freq)} 个不同的codons")

    # 2. 生成随机密文样本分析5位值频率
    print(f"   ✓ 生成 {num_samples} 个密文样本进行统计分析...")
    sample_ciphertexts = []
    for i in range(num_samples):
        # 生成随机数据
        random_data = bytes([random.randint(0, 255) for _ in range(16)])
        sample_ciphertexts.append(random_data)

    bit5_freq = analyze_5bit_frequency(sample_ciphertexts)

    # 3. 选择可用的codons(排除start/stop)
    available_codons = [(codon, freq) for codon, freq in codon_freq.items()
                        if codon not in START_STOP_CODONS]

    # 按频率排序,选择前32个
    available_codons.sort(key=lambda x: x[1], reverse=True)

    if len(available_codons) < 32:
        print(f"   ⚠ 警告: 只找到 {len(available_codons)} 个可用codons")
        # 补充缺失的codons
        used = set(c for c, _ in available_codons)
        for codon in ALL_CODONS:
            if codon not in used and codon not in START_STOP_CODONS:
                available_codons.append((codon, 0.001))
            if len(available_codons) >= 32:
                break

    selected_codons = available_codons[:32]

    # 4. 按频率匹配
    # 将5位值按频率排序
    bit5_sorted = sorted(bit5_freq.items(), key=lambda x: x[1], reverse=True)
    # 如果不足32个,补齐
    all_5bits = [format(i, '05b') for i in range(32)]
    existing_bits = set(b for b, _ in bit5_sorted)
    for bit in all_5bits:
        if bit not in existing_bits:
            bit5_sorted.append((bit, 0.001))

    # 创建映射表
    mapping_table = {}
    for (bit5, _), (codon, _) in zip(bit5_sorted[:32], selected_codons):
        mapping_table[bit5] = codon

    # 计算匹配度
    distance = 0
    for bit5, codon in mapping_table.items():
        bit_freq = bit5_freq.get(bit5, 0)
        codon_freq_val = codon_freq.get(codon, 0)
        distance += (bit_freq - codon_freq_val) ** 2
    distance = distance ** 0.5

    print(f"   ✓ 生成了32个codon映射")
    print(f"   ✓ 频率匹配度: {distance:.6f} (越小越好)")

    return mapping_table


def optimize_fake_ratio(encoded_dna: str, reference_dna: str, key: str,
                        test_ratios: List[float] = None) -> Tuple[float, Dict]:
    """
    优化fake codon的比例

    测试不同比例,返回最优方案
    """
    if test_ratios is None:
        test_ratios = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]

    print("\n🔍 优化Fake Codon比例...")
    print(f"   测试比例: {test_ratios}")

    reference_freq = analyze_codon_frequency(reference_dna)
    num_real_codons = len(encoded_dna) // 3

    results = []

    for ratio in test_ratios:
        # 生成该比例下的隐写DNA
        synthesized = embed_fake_data_with_ratio(
            encoded_dna, reference_freq, key, ratio
        )

        # 计算不可区分性
        synth_freq = analyze_codon_frequency(synthesized)
        distance = calculate_frequency_distance(synth_freq, reference_freq)

        results.append({
            'ratio': ratio,
            'distance': distance,
            'num_fake': len(synthesized) // 3 - num_real_codons,
            'synthesized_dna': synthesized
        })

        print(f"   比例 {ratio:.1f}: 距离={distance:.6f}, 伪codons={len(synthesized) // 3 - num_real_codons}")

    # 选择距离最小的
    best = min(results, key=lambda x: x['distance'])
    print(f"\n   ✅ 最优比例: {best['ratio']:.1f}")
    print(f"   ✅ 最小距离: {best['distance']:.6f}")

    return best['ratio'], best


def calculate_frequency_distance(freq1: Dict[str, float], freq2: Dict[str, float]) -> float:
    """计算两个频率分布的距离"""
    all_codons = set(freq1.keys()) | set(freq2.keys())
    distance = 0
    for codon in all_codons:
        diff = freq1.get(codon, 0) - freq2.get(codon, 0)
        distance += diff ** 2
    return distance ** 0.5


def embed_fake_data_with_ratio(encoded_dna: str, reference_freq: Dict[str, float],
                               key: str, ratio: float) -> str:
    """使用指定比例嵌入fake codons"""
    num_codons = len(encoded_dna) // 3
    num_fake = int(num_codons * ratio)

    # 获取已使用的codons
    used_codons = set()
    for i in range(0, len(encoded_dna), 3):
        used_codons.add(encoded_dna[i:i + 3])

    # 可用于fake的codons
    fake_codon_candidates = [(c, f) for c, f in reference_freq.items()
                             if c not in used_codons and c not in START_STOP_CODONS]

    if not fake_codon_candidates:
        # 如果没有可用的,从所有codons中选择
        fake_codon_candidates = [(c, reference_freq.get(c, 0.001))
                                 for c in ALL_CODONS
                                 if c not in START_STOP_CODONS]

    # 按频率分布选择fake codons
    total_freq = sum(f for _, f in fake_codon_candidates)
    if total_freq == 0:
        total_freq = 1

    fake_codons = []
    for codon, freq in fake_codon_candidates:
        count = int((freq / total_freq) * num_fake)
        fake_codons.extend([codon] * count)

    # 补齐或截断到目标数量
    while len(fake_codons) < num_fake:
        fake_codons.append(fake_codon_candidates[0][0])
    fake_codons = fake_codons[:num_fake]

    # 随机打乱
    random.seed(hashlib.md5((key + "fake").encode()).hexdigest())
    random.shuffle(fake_codons)

    # 确定插入位置
    total_codons = num_codons + num_fake
    positions = sorted(random.sample(range(total_codons), num_fake))

    # 插入fake codons
    real_codons = [encoded_dna[i:i + 3] for i in range(0, len(encoded_dna), 3)]
    synthesized = []
    fake_idx = 0
    real_idx = 0

    for i in range(total_codons):
        if fake_idx < len(positions) and i == positions[fake_idx]:
            synthesized.append(fake_codons[fake_idx])
            fake_idx += 1
        else:
            if real_idx < len(real_codons):
                synthesized.append(real_codons[real_idx])
            real_idx += 1

    return ''.join(synthesized)


def ciphertext_to_dna(ciphertext: bytes, mapping_table: Dict[str, str]) -> Tuple[str, int]:
    """使用映射表将密文转换为DNA"""
    binary = bytes_to_binary(ciphertext)
    padding_bits = (5 - len(binary) % 5) % 5
    binary += '0' * padding_bits

    dna = ''
    for i in range(0, len(binary), 5):
        chunk = binary[i:i + 5]
        codon = mapping_table[chunk]
        dna += codon

    return dna, padding_bits


def dna_to_ciphertext(dna: str, mapping_table: Dict[str, str], padding_bits: int) -> bytes:
    """使用映射表将DNA转换回密文"""
    reverse_mapping = {v: k for k, v in mapping_table.items()}
    binary = ''

    for i in range(0, len(dna), 3):
        codon = dna[i:i + 3]
        if codon in reverse_mapping:
            binary += reverse_mapping[codon]

    if padding_bits > 0:
        binary = binary[:-padding_bits]

    return binary_to_bytes(binary)


def remove_fake_data(synthesized_dna: str, mapping_table: Dict[str, str],
                     num_real_codons: int) -> str:
    """从合成DNA中移除fake codons"""
    reverse_mapping = {v: k for k, v in mapping_table.items()}
    all_codons = [synthesized_dna[i:i + 3] for i in range(0, len(synthesized_dna), 3)]

    # 识别真实codons(在映射表中的)
    real_codons = [c for c in all_codons if c in reverse_mapping]
    real_codons = real_codons[:num_real_codons]

    return ''.join(real_codons)


def encode_message(message_data: bytes, key: str, reference_dna: str,
                   auto_optimize: bool = True) -> Tuple[str, dict]:
    """
    完整编码过程

    参数:
        message_data: 要隐藏的消息
        key: 加密密钥
        reference_dna: 参考DNA序列
        auto_optimize: 是否自动优化fake ratio
    """
    print(f"\n{'=' * 70}")
    print("开始DNA隐写编码")
    print(f"{'=' * 70}")
    print(f"原始数据长度: {len(message_data)} 字节")

    # 步骤1: 生成最优映射表
    mapping_table = generate_mapping_table(reference_dna)

    # 步骤2: 加密
    print(f"\n🔐 加密消息...")
    ciphertext = aes_encrypt(message_data, key)
    print(f"   ✓ 密文长度: {len(ciphertext)} 字节")

    # 步骤3: 转换为DNA
    print(f"\n🧬 转换为DNA序列...")
    encoded_dna, padding_bits = ciphertext_to_dna(ciphertext, mapping_table)
    num_codons = len(encoded_dna) // 3
    print(f"   ✓ 编码DNA: {len(encoded_dna)} 碱基 ({num_codons} codons)")

    # 步骤4: 优化并嵌入fake Data
    if auto_optimize:
        best_ratio, best_result = optimize_fake_ratio(encoded_dna, reference_dna, key)
        synthesized_dna = best_result['synthesized_dna']
        fake_ratio = best_ratio
    else:
        # 使用默认比例0.7
        fake_ratio = 0.7
        reference_freq = analyze_codon_frequency(reference_dna)
        synthesized_dna = embed_fake_data_with_ratio(
            encoded_dna, reference_freq, key, fake_ratio
        )

    print(f"\n📊 最终结果:")
    print(f"   • 合成DNA长度: {len(synthesized_dna)} 碱基")
    print(f"   • 总codons: {len(synthesized_dna) // 3}")
    print(f"   • 真实codons: {num_codons}")
    print(f"   • 伪codons: {len(synthesized_dna) // 3 - num_codons}")
    print(f"   • Fake比例: {fake_ratio:.2f}")

    # 最终不可区分性分析
    synth_freq = analyze_codon_frequency(synthesized_dna)
    ref_freq = analyze_codon_frequency(reference_dna)
    final_distance = calculate_frequency_distance(synth_freq, ref_freq)
    print(f"   • 不可区分性: {final_distance:.6f}")

    metadata = {
        'padding_bits': padding_bits,
        'num_real_codons': num_codons,
        'ciphertext_length': len(ciphertext),
        'original_length': len(message_data),
        'mapping_table': mapping_table,
        'fake_ratio': fake_ratio
    }

    return synthesized_dna, metadata


def decode_message(synthesized_dna: str, key: str, metadata: dict) -> bytes:
    """完整解码过程"""
    print(f"\n{'=' * 70}")
    print("开始DNA隐写解码")
    print(f"{'=' * 70}")
    print(f"合成DNA长度: {len(synthesized_dna)} 碱基")

    # 步骤1: 移除fake Data
    print(f"\n🧹 移除伪数据...")
    encoded_dna = remove_fake_data(
        synthesized_dna,
        metadata['mapping_table'],
        metadata['num_real_codons']
    )
    print(f"   ✓ 恢复编码DNA: {len(encoded_dna)} 碱基")

    # 步骤2: 转换为密文
    print(f"\n🔄 转换为密文...")
    ciphertext = dna_to_ciphertext(
        encoded_dna,
        metadata['mapping_table'],
        metadata['padding_bits']
    )
    ciphertext = ciphertext[:metadata['ciphertext_length']]
    print(f"   ✓ 恢复密文: {len(ciphertext)} 字节")

    # 步骤3: 解密
    print(f"\n🔓 解密消息...")
    message_data = aes_decrypt(ciphertext, key)
    message_data = message_data[:metadata['original_length']]
    print(f"   ✓ 成功恢复: {len(message_data)} 字节")

    return message_data


def load_dna_file(filepath: str) -> str:
    """加载DNA文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.strip().split('\n')
    dna_sequence = ''
    for line in lines:
        if not line.startswith('>'):
            dna_sequence += ''.join(c.upper() for c in line if c.upper() in 'ACGT')
    return dna_sequence


def load_binary_file(filepath: str) -> bytes:
    """加载二进制文件"""
    with open(filepath, 'rb') as f:
        return f.read()


def save_stego_dna(filepath: str, dna: str, metadata: dict):
    """保存隐写DNA及元数据"""
    import json
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f">DNA-Courier Stego Sequence (Auto-Generated Mapping)\n")
        f.write(f">METADATA:padding_bits={metadata['padding_bits']}\n")
        f.write(f">METADATA:num_real_codons={metadata['num_real_codons']}\n")
        f.write(f">METADATA:ciphertext_length={metadata['ciphertext_length']}\n")
        f.write(f">METADATA:original_length={metadata['original_length']}\n")
        f.write(f">METADATA:fake_ratio={metadata['fake_ratio']:.4f}\n")

        # 保存映射表
        f.write(">MAPPING_TABLE_START\n")
        mapping_json = json.dumps(metadata['mapping_table'])
        f.write(f">{mapping_json}\n")
        f.write(">MAPPING_TABLE_END\n")

        # 写入DNA序列
        for i in range(0, len(dna), 70):
            f.write(dna[i:i + 70] + '\n')


def load_stego_dna(filepath: str) -> Tuple[str, dict]:
    """加载隐写DNA及元数据"""
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.strip().split('\n')
    metadata = {}
    dna = ''
    in_mapping = False
    mapping_json = ''

    for line in lines:
        if line.startswith('>MAPPING_TABLE_START'):
            in_mapping = True
            continue
        elif line.startswith('>MAPPING_TABLE_END'):
            in_mapping = False
            metadata['mapping_table'] = json.loads(mapping_json)
            mapping_json = ''
            continue
        elif in_mapping:
            if line.startswith('>'):
                mapping_json += line[1:]
            continue
        elif line.startswith('>METADATA:'):
            parts = line[10:].split('=')
            if len(parts) == 2:
                key, value = parts
                try:
                    if key == 'fake_ratio':
                        metadata[key] = float(value)
                    else:
                        metadata[key] = int(value)
                except:
                    metadata[key] = value
        elif line.startswith('>'):
            continue
        else:
            dna += ''.join(c.upper() for c in line if c.upper() in 'ACGT')

    return dna, metadata


def main():
    """
    主函数
    """
    # ========== 配置参数 ==========
    MODE = 'encode'  # 'encode' 或 'decode'
    DNA_FILE = r'.\4_Baselines\AAA36405.1.txt'
    SECRET_FILE = r'.\4_Baselines\secret.txt'
    OUTPUT_FILE = r'.\4_Baselines\output\CCRS\CCRS_AAA36405.1.txt'
    SECRET_KEY = r'.\4_Baselines\output\CCRS\my_secret_key_2024'
    AUTO_OPTIMIZE = True  # 是否自动优化fake ratio
    # =============================

    print("=" * 70)
    print("DNA隐写术工具 (改进版)")
    print("✨ 特性: 自动生成Mapping Table + 自动优化Fake Codon比例")
    print("=" * 70)

    if MODE == 'encode':
        print(f"\n📋 配置:")
        print(f"   模式: 编码")
        print(f"   参考DNA: {DNA_FILE}")
        print(f"   秘密文件: {SECRET_FILE}")
        print(f"   输出文件: {OUTPUT_FILE}")
        print(f"   自动优化: {'是' if AUTO_OPTIMIZE else '否'}")

        try:
            # 加载数据
            reference_dna = load_dna_file(DNA_FILE)
            print(f"\n✓ 加载参考DNA: {len(reference_dna)} 碱基")

            message_data = load_binary_file(SECRET_FILE)
            print(f"✓ 加载秘密消息: {len(message_data)} 字节")

            # 编码
            synthesized_dna, metadata = encode_message(
                message_data, SECRET_KEY, reference_dna, AUTO_OPTIMIZE
            )

            # 保存
            save_stego_dna(OUTPUT_FILE, synthesized_dna, metadata)
            print(f"\n💾 已保存到: {OUTPUT_FILE}")

            print(f"\n{'=' * 70}")
            print("✅ 编码完成!")
            print(f"{'=' * 70}")

        except FileNotFoundError as e:
            print(f"\n 文件未找到 - {e}")
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()

    elif MODE == 'decode':
        print(f"\n📋 配置:")
        print(f"   模式: 解码")
        print(f"   隐写DNA: {DNA_FILE}")
        print(f"   输出文件: {OUTPUT_FILE}")

        try:
            # 加载隐写DNA
            synthesized_dna, metadata = load_stego_dna(DNA_FILE)
            print(f"\n✓ 加载隐写DNA: {len(synthesized_dna)} 碱基")
            print(f"✓ 加载元数据: {len(metadata)} 项")

            # 解码
            message_data = decode_message(synthesized_dna, SECRET_KEY, metadata)

            # 保存
            with open(OUTPUT_FILE, 'wb') as f:
                f.write(message_data)
            print(f"\n💾 已保存到: {OUTPUT_FILE}")

            print(f"\n{'=' * 70}")
            print("✅ 解码完成!")
            print(f"{'=' * 70}")

        except FileNotFoundError as e:
            print(f"\n❌ 错误: 文件未找到 - {e}")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()