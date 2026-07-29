# main.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from utils import Vocabulary, Corpus, Generator, create_dummy_data
from lm import LM
import stega

# 配置
CONFIG = {
    'data_dir': './Data',
    'train_file': './Data/raw.txt',
    'test_file': './Data/test.txt',
    'model_path': './model.pth',

    # === 关键修改 ===
    'seq_len': 33,  # 198 / 6 = 33。这是 LSTM.txt 看到的序列长度
    # ================

    'batch_size': 32,
    'epochs': 50,
    'lr': 0.001,
    'embed_dim': 128,  # 词表变大了(4->4096)，建议稍微调大一点 embed_dim
    'hidden_dim': 256,  # 建议稍微调大隐藏层
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def train():
    # --- 修改部分开始 ---
    # 检查数据是否存在
    if not os.path.exists(CONFIG['train_file']):
        print("Error: Data file not found. Please run 'preprocess.py' first!")
        return None
    # --- 修改部分结束 ---

    print(">>> 2. Building Vocabulary...")
    # ... (其余代码保持不变) ...
    vocab = Vocabulary([CONFIG['train_file'], CONFIG['test_file']])
    print(f"Vocab size: {vocab.vocab_size}")

    train_corpus = Corpus(CONFIG['train_file'], vocab)
    train_gen = Generator(train_corpus.corpus, vocab)

    print(">>> 3. Initializing Model...")
    model = LM(vocab.vocab_size, CONFIG['embed_dim'], CONFIG['hidden_dim']).to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    # 忽略 padding 的 loss
    criterion = nn.NLLLoss(ignore_index=vocab.w2i.get('_PAD', 0))

    print(">>> 4. Training...")
    model.train()
    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        steps = 0
        # 增加 try-except 以防止某个 batch 出错中断训练
        try:
            for batch in train_gen.get_batch(CONFIG['batch_size']):
                batch = torch.LongTensor(batch).to(CONFIG['device'])
                input_seq = batch[:, :-1]
                target_seq = batch[:, 1:]

                optimizer.zero_grad()
                log_probs = model(input_seq)

                loss = criterion(log_probs.reshape(-1, vocab.vocab_size), target_seq.reshape(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1

                # 可选：如果数据量太大，限制每个 epoch 的步数
                if steps > 500: break
        except Exception as e:
            print(f"Warning: Error in batch: {e}")
            continue

        avg_loss = total_loss / steps if steps > 0 else 0
        print(f"Epoch {epoch + 1}/{CONFIG['epochs']}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), CONFIG['model_path'])
    print("Model saved.")
    return vocab

def test_steganography(vocab):
    print("\n>>> 5. Testing Steganography (ADG)...")
    model = LM(vocab.vocab_size, CONFIG['embed_dim'], CONFIG['hidden_dim']).to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['model_path']))

    # 随机生成秘密信息 (01 bit流)
    secret_msg = [random.randint(0, 1) for _ in range(50)]
    print(f"Secret Msg (first 10): {secret_msg[:10]}... Total len: {len(secret_msg)}")

    # Embedding
    # Context start with BOS
    context = [vocab.w2i['_BOS']]
    print("Embedding...")
    stega_seq_ids = stega.embed_secret(model, vocab, context, secret_msg, CONFIG['device'])

    # 将生成的ID转回 tokens
    stega_seq_tokens = [vocab.i2w.get(idx, '_UNK') for idx in stega_seq_ids]
    print(f"Generated Sequence: {' '.join(stega_seq_tokens)}")

    # Extraction
    # 完整的序列包括 BOS + 生成的部分
    full_seq_ids = context + stega_seq_ids
    print("Extracting...")
    extracted_msg = stega.extract_secret(model, vocab, full_seq_ids, CONFIG['device'])

    print(f"Extracted Msg (first 10): {extracted_msg[:10]}...")


    min_len = min(len(secret_msg), len(extracted_msg))
    acc = sum([1 for i in range(min_len) if secret_msg[i] == extracted_msg[i]]) / min_len
    print(f"Bit Error Rate: {1 - acc:.2%} (Accuracy: {acc:.2%})")


def generate_balanced_dataset(vocab, num_samples=8597):
    """
    生成长度严格对齐(198bp)的1:1正负样本集
    """
    print(f"\n>>> 正在准备生成对齐数据集 (目标: 正负各 {num_samples} 条, 长度: 198bp)...")

    # 1. 加载模型
    model = LM(vocab.vocab_size, CONFIG['embed_dim'], CONFIG['hidden_dim']).to(CONFIG['device'])
    if os.path.exists(CONFIG['model_path']):
        model.load_state_dict(torch.load(CONFIG['model_path']))
    model.eval()

    # 2. 读取 secret.txt 完整比特流
    with open('secret.txt', 'r', encoding='utf-8') as f:
        full_bits = [int(b) for b in f.read().strip() if b in ['0', '1']]

    # 3. 获取【已切分】的正样本
    # 注意：这里读取的是 train.txt，它是 process Data.py 处理后标准化的 198bp 片段
    pos_data = []
    if not os.path.exists(CONFIG['train_file']):
        print("错误：未找到 train.txt，请先运行 process Data.py")
        return

    with open(CONFIG['train_file'], 'r', encoding='utf-8') as f:
        # 跳过可能不完整的行，确保只取标准长度的数据
        lines = [line.strip() for line in f.readlines() if len(line.strip().split()) == 33]
        print("train.txt 中合法的 33-token 样本数 =", len(lines))
        pos_data = lines[:num_samples]

    actual_n = len(pos_data)
    if actual_n < num_samples:
        print(f"警告：正样本数量不足，仅能生成 {actual_n} 对。")

    # 4. 生成接力式隐藏的负样本
    neg_data = []
    global_bit_ptr = 0

    for i in range(actual_n):
        if global_bit_ptr >= len(full_bits):
            global_bit_ptr = 0  # 循环使用信息

        remaining_secret = full_bits[global_bit_ptr:]
        context = [vocab.w2i['_BOS']]

        # 调用 stega.py，target_len=33 严格对应 198bp
        stega_ids, bits_consumed = stega.embed_secret(
            model, vocab, context, remaining_secret, CONFIG['device'], target_len=33
        )

        global_bit_ptr += bits_consumed
        stega_tokens = [vocab.i2w.get(idx, '_UNK') for idx in stega_ids]
        neg_data.append(" ".join(stega_tokens))

    # 5. 保存结果
    os.makedirs('./data', exist_ok=True)
    pos_path = './data/positive_samples.txt'
    neg_path = './data/negative_samples.txt'

    with open(pos_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(pos_data))
    with open(neg_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(neg_data))

    print(f">>> 成功！正负样本长度均对齐为 {actual_n} 行，每行 33 个词 (198bp)。")

if __name__ == '__main__':
    vocab = train()
    if vocab:

        generate_balanced_dataset(vocab, num_samples=8597)


