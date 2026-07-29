import torch
import os
import numpy as np
import math
from utils import Vocabulary
from lm import LM
import stega

# ================= 配置区域 =================
CONFIG = {
    'train_file': './Data/train.txt',
    'test_file': './Data/test.txt',
    'model_path': './model.pth',
    'embed_dim': 128,
    'hidden_dim': 256,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # 输入的秘密文件
    'secret_file': r'.\3_Model_training & Stega_generating\adg_train\secret.txt',
    # 输出的隐写DNA文件
    'output_dna': './stego_dna.txt',
    # (可选) 验证提取出的文件
    'extracted_file': 'recovered_secret.out'
}


# ===========================================

def file_to_bits(file_path):
    """读取二进制文件并转换为 01 比特流"""
    bits = []
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'rb') as f:
        content = f.read()
        for byte in content:
            # 将每个字节转换为8位二进制 (例如 255 -> 11111111)
            bits.extend([int(b) for b in f"{byte:08b}"])
    return bits, len(content)


def bits_to_file(bits, output_path):
    """将比特流转换回二进制文件"""
    byte_data = bytearray()
    for i in range(0, len(bits), 8):
        byte_chunk = bits[i:i + 8]
        if len(byte_chunk) < 8: break  # 丢弃最后不足8位的部分

        byte_val = int("".join(str(x) for x in byte_chunk), 2)
        byte_data.append(byte_val)

    with open(output_path, 'wb') as f:
        f.write(byte_data)


def main():
    print(">>> 1. Environment Setup...")
    # 检查秘密文件
    if not os.path.exists(CONFIG['secret_file']):
        print(f" Error: '{CONFIG['secret_file']}' not found!")
        print("Please create this file or put your secret Data in it.")
        return

    # 重建词表
    if not os.path.exists(CONFIG['train_file']):

        return
    vocab = Vocabulary([CONFIG['train_file'], CONFIG['test_file']])

    # 加载模型
    print(">>> 2. Loading Model...")
    if not os.path.exists(CONFIG['model_path']):
        print(" Error: Model not found.")
        return

    model = LM(vocab.vocab_size, CONFIG['embed_dim'], CONFIG['hidden_dim']).to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
    model.eval()

    # ---------------------------------------------------------
    # 3. 读取秘密文件
    # ---------------------------------------------------------
    print(f"\n>>> 3. Reading secret file: {CONFIG['secret_file']}")
    secret_bits, file_size = file_to_bits(CONFIG['secret_file'])
    print(f"    File Size: {file_size} bytes")
    print(f"    Total Bits: {len(secret_bits)} bits")

    # ---------------------------------------------------------
    # 4. 执行隐写嵌入 (Embedding)
    # ---------------------------------------------------------
    print("\n>>> 4. Embedding into DNA Sequence...")
    context = [vocab.w2i['_BOS']]

    try:
        # 调用 stega 嵌入
        stega_ids = stega.embed_secret(model, vocab, context, secret_bits, CONFIG['device'])
    except Exception as e:
        print(f" Error during embedding: {e}")
        return

    # 转换回 Token (6-mer)
    stega_tokens = [vocab.i2w.get(idx, '_UNK') for idx in stega_ids]
    valid_tokens = [t for t in stega_tokens if not t.startswith('_')]

    # 拼接 DNA 序列 (不带空格)
    final_dna_sequence = "".join(valid_tokens)

    # 计算容量效率 (BPB: Bit Per Base)
    bpb = len(secret_bits) / len(final_dna_sequence) if final_dna_sequence else 0

    print("\n" + "=" * 40)
    print(" GENERATION COMPLETE")
    print("=" * 40)
    print(f"Target Secret: {CONFIG['secret_file']}")
    print(f"DNA Length:    {len(final_dna_sequence)} bp")
    print(f"Embedding Rate:{bpb:.4f} bits/base")
    print("-" * 40)

    # 保存结果
    with open(CONFIG['output_dna'], "w") as f:
        f.write(final_dna_sequence)
    print(f"DNA Sequence saved to: {CONFIG['output_dna']}")

    # ---------------------------------------------------------
    # 5. 验证提取 (Self-Verification)
    # ---------------------------------------------------------
    print("\n>>> 5. Verifying (Extracting back to file)...")

    # 模拟提取过程
    full_seq_ids = [vocab.w2i['_BOS']] + stega_ids
    extracted_bits = stega.extract_secret(model, vocab, full_seq_ids, CONFIG['device'])

    # 截取有效长度
    valid_len = len(secret_bits)
    if len(extracted_bits) < valid_len:

        recovered_bits = extracted_bits
    else:
        recovered_bits = extracted_bits[:valid_len]

    # 计算误码率
    errors = sum(1 for i in range(len(recovered_bits)) if recovered_bits[i] != secret_bits[i])
    ber = errors / len(secret_bits)
    print(f"    Bit Error Rate (BER): {ber:.2%}")

    # 保存提取出的文件
    bits_to_file(recovered_bits, CONFIG['extracted_file'])
    print(f"    Recovered file saved to: {CONFIG['extracted_file']}")

    if ber == 0:
        print(" Perfect Reconstruction! The files are identical.")
    else:
        print(" Data corrupted during steganography.")


if __name__ == "__main__":
    main()