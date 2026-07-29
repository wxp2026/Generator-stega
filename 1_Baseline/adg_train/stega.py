# stega.py
import torch
import numpy as np


def bits_to_int(bits):
    """二进制列表转整数"""
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def int_to_bits(n, length):
    """整数转二进制列表"""
    return [int(x) for x in f"{n:0{length}b}"]


def adg_coding(probs, vocab, secret_bits, mode='hide'):
    """
    Adaptive Dynamic Grouping (ADG) 核心逻辑 - 修复版
    """
    # 1. 排序概率
    probs = probs.cpu().detach().numpy().flatten()

    # 过滤掉特殊token
    # 过滤掉特殊token
    mask_indices = [vocab.w2i.get(k) for k in ['_PAD', '_UNK', '_BOS', '_EOS']]
    for idx in mask_indices:
        if idx is not None:
            probs[idx] = 0

    # 安全检查：如果总概率太小（几乎全是0），直接跳过
    if probs.sum() < 1e-9:
        return None, 0

    probs = probs / probs.sum()  # 重新归一化

    # 获取排序后的索引
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    p_max = sorted_probs[0]

    # 2. 确定嵌入容量 Bs
    if p_max > 0.5:
        return None, 0

    b_s = int(np.ceil(-np.log2(p_max)))
    num_groups = 2 ** b_s

    # 3. 自适应分组 (ADG)
    groups = [[] for _ in range(num_groups)]
    # ... (此处省略未改动的初始化代码) ...

    # --- 分组逻辑 ---
    final_groups_tokens = [[] for _ in range(num_groups)]
    final_groups_probs = [0.0] * num_groups

    target_mean = 1.0 / num_groups
    current_group_id = 0

    token_map = {}

    available_tokens = list(sorted_indices)

    for t_idx in available_tokens:
        if probs[t_idx] < 1e-10: continue  # 忽略概率极小的词

        final_groups_tokens[current_group_id].append(t_idx)
        final_groups_probs[current_group_id] += probs[t_idx]
        token_map[t_idx] = current_group_id

        # 切换到下一组
        if final_groups_probs[current_group_id] >= target_mean and current_group_id < num_groups - 1:
            current_group_id += 1

    # 4. 执行嵌入或提取
    if mode == 'hide':
        if len(secret_bits) < b_s:
            bits_to_hide = secret_bits + [0] * (b_s - len(secret_bits))
        else:
            bits_to_hide = secret_bits[:b_s]

        group_idx = bits_to_int(bits_to_hide)

        # [Fix] 获取候选词
        candidate_tokens = final_groups_tokens[group_idx]

        # [Fix] 安全检查：如果选中的组为空，或者组内概率和为0，视为嵌入失败
        if not candidate_tokens:
            return None, 0

        candidate_probs = [probs[t] for t in candidate_tokens]
        prob_sum = sum(candidate_probs)

        if prob_sum < 1e-10:  # 防止除以0
            return None, 0

        candidate_probs = np.array(candidate_probs) / prob_sum  # 安全归一化

        selected_token = np.random.choice(candidate_tokens, p=candidate_probs)
        return selected_token, b_s

    elif mode == 'extract':
        target_token = secret_bits

        if target_token not in token_map:
            return [], 0

        group_idx = token_map[target_token]
        extracted_bits = int_to_bits(group_idx, b_s)
        return extracted_bits, b_s


def embed_secret(model, vocab, context_tokens, secret_msg_bits, device, target_len=33):
    model.eval()
    generated_ids = []
    bit_ptr = 0

    curr_seq = torch.LongTensor([context_tokens]).to(device)

    for _ in range(target_len):
        with torch.no_grad():
            probs = model.get_prob(curr_seq)[0]

        # 额外屏蔽 _EOS，防止 fallback / argmax 选中它
        safe_probs = probs.clone()
        eos_id = vocab.w2i.get('_EOS')
        if eos_id is not None:
            safe_probs[eos_id] = 0

        if bit_ptr < len(secret_msg_bits):
            token_id, bits_used = adg_coding(safe_probs, vocab, secret_msg_bits[bit_ptr:], mode='hide')
            if token_id is None:
                token_id = torch.argmax(safe_probs).item()
                bits_used = 0
            bit_ptr += bits_used
        else:
            token_id = torch.argmax(safe_probs).item()

        generated_ids.append(token_id)
        next_in = torch.LongTensor([[token_id]]).to(device)
        curr_seq = torch.cat([curr_seq, next_in], dim=1)

    return generated_ids, bit_ptr



def extract_secret(model, vocab, full_seq_ids, device):
    """提取主循环"""
    model.eval()
    extracted_bits = []

    # 初始context是BOS
    # full_seq_ids 包含 [BOS, t1, t2, ..., EOS]
    # 我们需要逐步输入 [BOS] -> 预测t1 -> 归属组 -> 提取bits

    curr_seq = torch.LongTensor([[full_seq_ids[0]]]).to(device)

    for i in range(1, len(full_seq_ids)):
        true_token = full_seq_ids[i]

        with torch.no_grad():
            probs = model.get_prob(curr_seq)[0]

        bits, bits_len = adg_coding(probs, vocab, true_token, mode='extract')

        if bits_len > 0:
            extracted_bits.extend(bits)

        # 更新序列
        next_in = torch.LongTensor([[true_token]]).to(device)
        curr_seq = torch.cat([curr_seq, next_in], dim=1)

        if true_token == vocab.w2i['_EOS']: break

    return extracted_bits