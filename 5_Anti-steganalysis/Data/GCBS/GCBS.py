
import random
from typing import Tuple, List

# ==================== 常量定义 ====================

# DNA碱基到二进制映射 (Figure 3)
DNA_TO_BINARY = {'A': '00', 'C': '01', 'G': '10', 'T': '11'}
BINARY_TO_DNA = {'00': 'A', '01': 'C', '10': 'G', '11': 'T'}

# 通用互补规则 (Figure 4)
GENERIC_COMPLEMENT = {'A': 'C', 'C': 'T', 'G': 'A', 'T': 'G'}

# 密码子到字符映射 (Table 1)
CODON_TO_CHAR = {
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAA': 'B', 'TGA': 'B', 'TAG': 'B',
    'TGT': 'C', 'TGC': 'C',
    'GAT': 'D', 'GAC': 'D',
    'GAA': 'E', 'GAG': 'E',
    'TTT': 'F', 'TTC': 'F',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    'CAT': 'H', 'CAC': 'H',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
    'AAG': 'K', 'AAA': 'K',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATG': 'M',
    'AAT': 'N', 'AAC': 'N',
    'TTA': 'O', 'TTG': 'O',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AGA': 'U', 'AGG': 'U',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TGG': 'W',
    'AGT': 'X', 'AGC': 'X',
    'TAT': 'Y',
    'TAC': 'Z',
}

# 字符到密码子的映射（取第一个）
CHAR_TO_CODONS = {}
for codon, char in CODON_TO_CHAR.items():
    if char not in CHAR_TO_CODONS:
        CHAR_TO_CODONS[char] = []
    CHAR_TO_CODONS[char].append(codon)


# ==================== 基础转换函数 ====================

def binary_to_dna(binary_str: str) -> str:
    """二进制字符串转DNA序列"""
    if len(binary_str) % 2 != 0:
        binary_str = '0' + binary_str
    dna = ''
    for i in range(0, len(binary_str), 2):
        dna += BINARY_TO_DNA[binary_str[i:i + 2]]
    return dna


def dna_to_binary(dna_str: str) -> str:
    """DNA序列转二进制字符串"""
    return ''.join(DNA_TO_BINARY[base] for base in dna_str)


def text_to_binary(text: str) -> str:
    """文本转二进制"""
    return ''.join(format(ord(char), '08b') for char in text)


def binary_to_text(binary_str: str) -> str:
    """二进制转文本"""
    while len(binary_str) % 8 != 0:
        binary_str = '0' + binary_str
    text = ''
    for i in range(0, len(binary_str), 8):
        byte = binary_str[i:i + 8]
        text += chr(int(byte, 2))
    return text


def get_complement(base: str, times: int = 1) -> str:
    """获取碱基的n次互补"""
    result = base
    for _ in range(times):
        result = GENERIC_COMPLEMENT[result]
    return result




def create_playfair_matrix(key: str) -> List[List[str]]:
    """创建5x5 Playfair矩阵"""
    key = key.upper()
    seen = set()
    key_chars = []

    for c in key:
        if c not in seen and c.isalpha():
            seen.add(c)
            key_chars.append(c)

    alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'  # 无J
    for c in alphabet:
        if c not in seen:
            key_chars.append(c)

    return [key_chars[i * 5:(i + 1) * 5] for i in range(5)]


def find_position(matrix: List[List[str]], char: str) -> Tuple[int, int]:
    """在Playfair矩阵中查找字符位置"""
    char = char.upper()
    if char == 'J':
        char = 'I'
    for i in range(5):
        for j in range(5):
            if matrix[i][j] == char:
                return (i, j)
    return (-1, -1)


def playfair_encrypt_pair(matrix: List[List[str]], a: str, b: str) -> str:
    """Playfair加密一对字符"""
    row_a, col_a = find_position(matrix, a)
    row_b, col_b = find_position(matrix, b)

    if row_a == row_b:
        return matrix[row_a][(col_a + 1) % 5] + matrix[row_b][(col_b + 1) % 5]
    elif col_a == col_b:
        return matrix[(row_a + 1) % 5][col_a] + matrix[(row_b + 1) % 5][col_b]
    else:
        return matrix[row_a][col_b] + matrix[row_b][col_a]


def playfair_decrypt_pair(matrix: List[List[str]], a: str, b: str) -> str:
    """Playfair解密一对字符"""
    row_a, col_a = find_position(matrix, a)
    row_b, col_b = find_position(matrix, b)

    if row_a == row_b:
        return matrix[row_a][(col_a - 1) % 5] + matrix[row_b][(col_b - 1) % 5]
    elif col_a == col_b:
        return matrix[(row_a - 1) % 5][col_a] + matrix[(row_b - 1) % 5][col_b]
    else:
        return matrix[row_a][col_b] + matrix[row_b][col_a]


# ==================== DNA-氨基酸转换 ====================

def dna_to_amino_with_ambiguity(dna_seq: str) -> Tuple[str, str]:
    """
    Step 2: 将DNA转换为氨基酸链，记录ambiguity
    返回: (氨基酸链, ambiguity二进制字符串)
    """
    # 填充至3的倍数
    padding = (3 - len(dna_seq) % 3) % 3
    padded_dna = dna_seq + 'A' * padding

    amino_chain = ''
    ambiguity_bits = ''

    for i in range(0, len(padded_dna), 3):
        codon = padded_dna[i:i + 3]

        if codon in CODON_TO_CHAR:
            char = CODON_TO_CHAR[codon]
            amino_chain += char

            # 记录使用了哪个密码子变体（用2位）
            possible_codons = CHAR_TO_CODONS.get(char, ['AAA'])
            if codon in possible_codons:
                idx = possible_codons.index(codon)
            else:
                idx = 0
            ambiguity_bits += format(idx, '02b')
        else:
            amino_chain += 'A'
            ambiguity_bits += '00'

    return amino_chain, ambiguity_bits


def amino_to_dna_with_ambiguity(amino_chain: str, ambiguity_bits: str) -> str:
    """
    Step 4: 使用ambiguity将氨基酸链转回DNA
    """
    dna = ''

    for i, char in enumerate(amino_chain):
        if char in CHAR_TO_CODONS:
            possible_codons = CHAR_TO_CODONS[char]

            # 从ambiguity中读取索引
            bit_pos = i * 2
            if bit_pos + 2 <= len(ambiguity_bits):
                idx = int(ambiguity_bits[bit_pos:bit_pos + 2], 2)
                if idx < len(possible_codons):
                    dna += possible_codons[idx]
                else:
                    dna += possible_codons[0]
            else:
                dna += possible_codons[0]
        else:
            dna += 'AAA'

    return dna


# ==================== DNA Playfair加密/解密 ====================

def dna_playfair_encrypt(message_dna: str, key: str) -> Tuple[str, str, int]:
    """
    Algorithm 1 Steps 1-4: DNA Playfair加密
    返回: (加密后的DNA, ambiguity二进制串, ambiguity长度)
    """
    # Step 2: DNA转氨基酸链
    amino_chain, ambiguity_bits = dna_to_amino_with_ambiguity(message_dna)

    # 确保氨基酸链长度为偶数（Playfair需要）
    if len(amino_chain) % 2 != 0:
        amino_chain += 'X'

    # Step 3: Playfair加密
    matrix = create_playfair_matrix(key)
    encrypted_amino = ''

    for i in range(0, len(amino_chain), 2):
        encrypted_amino += playfair_encrypt_pair(matrix, amino_chain[i], amino_chain[i + 1])

    # Step 4: 加密的氨基酸链转回DNA（这里不使用原始ambiguity）
    # 为了简化，我们使用默认的第一个密码子
    encrypted_dna = ''
    for char in encrypted_amino:
        if char in CHAR_TO_CODONS:
            encrypted_dna += CHAR_TO_CODONS[char][0]
        else:
            encrypted_dna += 'AAA'

    return encrypted_dna, ambiguity_bits, len(ambiguity_bits)


def dna_playfair_decrypt(encrypted_dna: str, ambiguity_bits: str, key: str) -> str:
    """
    Algorithm 2: DNA Playfair解密
    """
    # 将加密的DNA转为氨基酸链
    amino_chain, _ = dna_to_amino_with_ambiguity(encrypted_dna)

    # Playfair解密
    matrix = create_playfair_matrix(key)
    decrypted_amino = ''

    for i in range(0, len(amino_chain) - 1, 2):
        decrypted_amino += playfair_decrypt_pair(matrix, amino_chain[i], amino_chain[i + 1])

    # 处理奇数长度
    if len(amino_chain) % 2 != 0:
        decrypted_amino += amino_chain[-1]

    # 使用ambiguity转回DNA
    decrypted_dna = amino_to_dna_with_ambiguity(decrypted_amino, ambiguity_bits)

    return decrypted_dna


# ==================== 回文检测 ====================

def is_palindrome(dna_seq: str) -> bool:
    """检查是否为Watson-Crick回文"""
    wc_complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    complement_reverse = ''.join(wc_complement.get(b, 'A') for b in reversed(dna_seq))
    return dna_seq == complement_reverse


def find_shortest_palindrome(dna_seq: str) -> str:
    """
    Step 5.1: 在cover DNA中查找最短回文
    """
    for length in range(4, min(21, len(dna_seq) + 1)):
        for i in range(len(dna_seq) - length + 1):
            subseq = dna_seq[i:i + length]
            if is_palindrome(subseq):
                return subseq
    return 'AATT'  # 默认回文



def gcbs_substitute(cover_dna: str, message_dna: str, key: str) -> str:

    random.seed(hash(key))
    positions = list(range(len(cover_dna)))
    random.shuffle(positions)

    stego_dna = list(cover_dna)

    for i, msg_base in enumerate(message_dna):
        if i >= len(positions):
            break

        pos = positions[i]
        cover_base = cover_dna[pos]

        # GCBS规则
        if msg_base == 'A':
            stego_dna[pos] = cover_base
        elif msg_base == 'C':
            stego_dna[pos] = get_complement(cover_base, 1)
        elif msg_base == 'G':
            stego_dna[pos] = get_complement(cover_base, 2)
        elif msg_base == 'T':
            stego_dna[pos] = get_complement(cover_base, 3)

    return ''.join(stego_dna)


def gcbs_extract(stego_dna: str, cover_dna: str, key: str) -> str:

    random.seed(hash(key))
    positions = list(range(len(cover_dna)))
    random.shuffle(positions)

    message_dna = ''

    for pos in positions:
        if pos >= len(stego_dna) or pos >= len(cover_dna):
            break

        stego_base = stego_dna[pos]
        cover_base = cover_dna[pos]

        # 反向GCBS
        if stego_base == cover_base:
            message_dna += 'A'
        elif stego_base == get_complement(cover_base, 1):
            message_dna += 'C'
        elif stego_base == get_complement(cover_base, 2):
            message_dna += 'G'
        elif stego_base == get_complement(cover_base, 3):
            message_dna += 'T'
        else:
            message_dna += 'A'

    return message_dna



def generate_random_segments(total_length: int, key: str, salt: str) -> List[int]:
    """
    严格按照Algorithm 1 Step 6.2/6.3生成随机片段长度

    Args:
        total_length: 总长度
        key: 密钥
        salt: 盐值（用于区分r和k序列）

    Returns:
        长度列表，总和等于total_length
    """
    random.seed(hash(key + salt))

    segments = []
    remaining = total_length

    while remaining > 0:
        # 生成1到min(20, remaining)之间的随机长度
        max_seg = min(20, remaining)
        if max_seg == 1:
            seg_len = 1
        else:
            seg_len = random.randint(1, max_seg)

        segments.append(seg_len)
        remaining -= seg_len

    return segments


def insertion_embed(stego_dna: str, cover_dna: str, key: str) -> str:

    # Step 6.2: 生成r序列（用于stego_dna）
    r_lengths = generate_random_segments(len(stego_dna), key, "_r")


    k_lengths = generate_random_segments(len(cover_dna), key, "_k")


    stego_segments = []
    pos = 0
    for length in r_lengths:
        stego_segments.append(stego_dna[pos:pos + length])
        pos += length

    cover_segments = []
    pos = 0
    for length in k_lengths:
        cover_segments.append(cover_dna[pos:pos + length])
        pos += length


    result = ''
    max_segments = max(len(stego_segments), len(cover_segments))

    for i in range(max_segments):
        # 先添加stego片段（r序列）
        if i < len(stego_segments):
            result += stego_segments[i]
        # 再添加cover片段（k序列）
        if i < len(cover_segments):
            result += cover_segments[i]

    return result


def insertion_extract(combined_dna: str, key: str) -> Tuple[str, str]:

    half_len = len(combined_dna) // 2

    # 重新生成相同的随机序列
    r_lengths = generate_random_segments(half_len, key, "_r")
    k_lengths = generate_random_segments(half_len, key, "_k")

    # 反向分离
    stego_dna = ''
    cover_dna = ''
    pos = 0
    max_segments = max(len(r_lengths), len(k_lengths))

    for i in range(max_segments):
        # 提取stego片段
        if i < len(r_lengths) and pos < len(combined_dna):
            stego_dna += combined_dna[pos:pos + r_lengths[i]]
            pos += r_lengths[i]

        # 提取cover片段
        if i < len(k_lengths) and pos < len(combined_dna):
            cover_dna += combined_dna[pos:pos + k_lengths[i]]
            pos += k_lengths[i]

    return stego_dna, cover_dna



def embed_message(cover_dna: str, message: str, key: str) -> str:

    print(f"原始消息: '{message}' ({len(message)} 字符)")
    print(f"Cover DNA长度: {len(cover_dna)} bases")

    # Step 1: 将消息转换为DNA
    binary_msg = text_to_binary(message)
    message_dna = binary_to_dna(binary_msg)
    print(f"Step 1: 消息转DNA: {len(message_dna)} bases")

    # Steps 2-4: Playfair加密
    encrypted_dna, ambiguity_bits, amb_len = dna_playfair_encrypt(message_dna, key)
    print(f"Steps 2-4: Playfair加密完成")
    print(f"  - 加密DNA长度: {len(encrypted_dna)} bases")
    print(f"  - Ambiguity长度: {amb_len} bits")

    # 创建完整消息：Header + Ambiguity + Encrypted
    # Header: 16位存储ambiguity长度
    header = format(amb_len, '016b')
    full_payload = header + ambiguity_bits + dna_to_binary(encrypted_dna)
    payload_dna = binary_to_dna(full_payload)

    print(f"Step 4+: 完整Payload: {len(payload_dna)} bases")
    print(f"  - Header: 8 bases (16 bits)")
    print(f"  - Ambiguity: {len(binary_to_dna(ambiguity_bits))} bases")
    print(f"  - Encrypted: {len(encrypted_dna)} bases")

    # Step 5: Substitution Phase
    # Step 5.1-5.2: 查找回文并添加end signal
    palindrome = find_shortest_palindrome(cover_dna)
    end_signal = 'T' + palindrome + 'T'
    print(f"Step 5.1-5.2: 回文end signal: {end_signal}")

    message_with_signal = payload_dna + end_signal

    # 检查容量
    if len(message_with_signal) > len(cover_dna):
        raise ValueError(f"消息过长！需要{len(message_with_signal)} bases，仅有{len(cover_dna)} bases")

    # Step 5.3-5.4: GCBS替换
    stego_dna = gcbs_substitute(cover_dna, message_with_signal, key)

    # Step 5.5: 截断至原长度
    stego_dna = stego_dna[:len(cover_dna)]
    print(f"Step 5: GCBS替换完成，Stego长度: {len(stego_dna)} bases")

    # Step 6: Insertion Phase
    final_dna = insertion_embed(stego_dna, cover_dna, key)
    print(f"Step 6: Insertion完成，最终长度: {len(final_dna)} bases")

    capacity_used = (len(message_with_signal) / len(cover_dna)) * 100
    print(f"容量使用率: {capacity_used:.2f}%")

    return final_dna


def extract_message(stego_dna: str, key: str) -> str:
    """
    Algorithm 2: The Extraction Process (完整实现)
    """
    print(f"Stego DNA长度: {len(stego_dna)} bases")

    # Step 1: Reference Recovery Phase
    extracted_stego, cover_dna = insertion_extract(stego_dna, key)
    print(f"Step 1: Insertion反向完成")
    print(f"  - 提取的Stego: {len(extracted_stego)} bases")
    print(f"  - 恢复的Cover: {len(cover_dna)} bases")

    # Step 2: Message Recovery Phase
    # Step 2.1: 反向GCBS
    full_message_dna = gcbs_extract(extracted_stego, cover_dna, key)

    # Step 2.2: 查找end signal
    palindrome = find_shortest_palindrome(cover_dna)
    end_signal = 'T' + palindrome + 'T'

    signal_pos = full_message_dna.find(end_signal)
    if signal_pos == -1:
        print("警告: 未找到end signal")
        signal_pos = len(full_message_dna)
    else:
        print(f"Step 2.2: 在位置{signal_pos}找到end signal")

    # 提取payload（不含end signal）
    payload_dna = full_message_dna[:signal_pos]
    payload_bits = dna_to_binary(payload_dna)

    # 解析Header（前16位）
    if len(payload_bits) < 16:
        raise ValueError("Payload太短，无法解析Header")

    amb_len = int(payload_bits[:16], 2)
    print(f"Step 2.3: 从Header读取Ambiguity长度: {amb_len} bits")

    # 分离ambiguity和encrypted
    header_end = 16
    amb_end = header_end + amb_len

    if amb_end > len(payload_bits):
        print(f"警告: Ambiguity长度异常，使用剩余数据")
        ambiguity_bits = payload_bits[header_end:]
        encrypted_bits = ''
    else:
        ambiguity_bits = payload_bits[header_end:amb_end]
        encrypted_bits = payload_bits[amb_end:]

    encrypted_dna = binary_to_dna(encrypted_bits)

    print(f"Step 2.4: 数据分离完成")
    print(f"  - Ambiguity: {len(ambiguity_bits)} bits")
    print(f"  - Encrypted DNA: {len(encrypted_dna)} bases")

    # Step 3: Playfair解密
    decrypted_dna = dna_playfair_decrypt(encrypted_dna, ambiguity_bits, key)
    print(f"Step 3: Playfair解密完成，DNA长度: {len(decrypted_dna)} bases")

    # Step 4: 转回文本
    decrypted_bits = dna_to_binary(decrypted_dna)
    message = binary_to_text(decrypted_bits)

    # 清理不可打印字符
    message = ''.join(c for c in message if c.isprintable() or c in '\n\t\r')

    return message


# ==================== 主程序 ====================

def main():

    DNA_INPUT = r".\4_Baselines\AAA03751.1.txt"
    SECRET_INPUT = r".\4_Baselines\half secret.txt"
    STEGO_OUTPUT = r".\4_Baselines\output\GCBS\GCBS_AAA03751.1.txt"
    EXTRACTED_OUTPUT = "extracted_message.txt"
    SECRET_KEY = "EGYPT"
    TEST_EXTRACTION = True
    # =========================

    print("=" * 70)
    print("DNA隐写术 - 严格按照论文Algorithm 1 & 2实现")
    print("Khalifa et al. (2016)")
    print("=" * 70)

    # ========== 嵌入 ==========
    print("\n【嵌入阶段 - Algorithm 1】")
    print("-" * 70)

    try:
        # 读取DNA
        with open(DNA_INPUT, 'r', encoding='utf-8') as f:
            dna_content = f.read()

        dna_sequence = ''
        for line in dna_content.strip().split('\n'):
            if not line.startswith('>'):
                dna_sequence += ''.join(c.upper() for c in line if c.upper() in 'ACGT')

        print(f"✓ 读取Cover DNA: {len(dna_sequence)} bases")

        # 读取消息
        with open(SECRET_INPUT, 'r', encoding='utf-8') as f:
            secret_message = f.read()

        print(f"✓ 读取消息: {len(secret_message)} 字符")
        print(f"✓ 密钥: {SECRET_KEY}\n")

        # 执行嵌入
        stego_dna = embed_message(dna_sequence, secret_message, SECRET_KEY)

        # 保存
        with open(STEGO_OUTPUT, 'w') as f:
            f.write(">Stego-DNA\n")
            for i in range(0, len(stego_dna), 70):
                f.write(stego_dna[i:i + 70] + '\n')

        print(f"\n✓ Stego DNA已保存: {STEGO_OUTPUT}")

    except Exception as e:
        print(f"✗ 嵌入错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== 提取 ==========
    if TEST_EXTRACTION:
        print("\n" + "=" * 70)
        print("【提取阶段 - Algorithm 2 (盲提取)】")
        print("-" * 70)

        try:
            # 读取stego
            with open(STEGO_OUTPUT, 'r') as f:
                stego_content = f.read()

            stego_sequence = ''
            for line in stego_content.strip().split('\n'):
                if not line.startswith('>'):
                    stego_sequence += ''.join(c.upper() for c in line if c.upper() in 'ACGT')

            print(f"✓ 读取Stego DNA: {len(stego_sequence)} bases")
            print(f"✓ 密钥: {SECRET_KEY}\n")

            # 执行提取
            extracted = extract_message(stego_sequence, SECRET_KEY)

            # 保存
            with open(EXTRACTED_OUTPUT, 'w', encoding='utf-8') as f:
                f.write(extracted)

            print(f"\n✓ 提取的消息: '{extracted}'")
            print(f"✓ 已保存: {EXTRACTED_OUTPUT}")

            # 验证
            print("\n【验证】")
            print("-" * 70)
            print(f"原始: '{secret_message}'")
            print(f"提取: '{extracted}'")

            if secret_message.strip() == extracted.strip():
                print("✓✓✓ 验证成功！消息完全匹配！")
            else:
                print("✗ 警告: 消息不完全匹配")

        except Exception as e:
            print(f"✗ 提取错误: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()