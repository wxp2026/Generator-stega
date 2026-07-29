#!/usr/bin/env python3
"""
隐写核心模块 - 编码器和解码器共享（修正版）

修正点：
1. 编码器/解码器继续共享同一套 Canonical Huffman 码构造逻辑。
2. 修复“尾部剩余 bit 不足以形成完整码字”时的非对称问题：
   - 原实现会直接回退到最高概率 token 且记为 0 bit，
     但解码端仍会把该 token 按 Huffman 码提取，导致尾部污染。
   - 新实现会在 secret 到达末尾但尚未匹配完整码字时，选择一个
     “以当前剩余 bit 为前缀”的合法 Huffman 码字对应 token。
     解码端仍按正常码字提取，再由 remaining_bits 截断到 payload 长度。
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import heapq


# ==================== 基础工具函数 ====================

def bits2int(bits: List[int]) -> int:
    value = 0
    for i, b in enumerate(bits):
        value += (1 << i) * int(b)
    return value


def int2bits(value: int, bit_len: int) -> List[int]:
    return [(value >> i) & 1 for i in range(bit_len)]


def near(alist: List[float], anum: float) -> int:
    if len(alist) == 0:
        return 0
    if anum <= alist[0]:
        return 0
    if anum >= alist[-1]:
        return len(alist) - 1

    bottom = 0
    up = len(alist) - 1
    while up - bottom > 1:
        mid = (up + bottom) // 2
        if alist[mid] < anum:
            bottom = mid
        elif alist[mid] > anum:
            up = mid
        else:
            return mid

    if alist[bottom] - anum < anum - alist[up]:
        return bottom
    return up


# ==================== Huffman / Canonical Huffman ====================

class _HuffNode:
    __slots__ = ("p", "token_id", "left", "right")

    def __init__(
        self,
        p: float,
        token_id: Optional[int] = None,
        left: Optional['_HuffNode'] = None,
        right: Optional['_HuffNode'] = None,
    ):
        self.p = float(p)
        self.token_id = token_id
        self.left = left
        self.right = right


def _normalize_probs(prob_list: List[float]) -> List[float]:
    total = float(sum(prob_list))
    if total <= 0:
        n = max(1, len(prob_list))
        return [1.0 / n] * n
    return [float(p) / total for p in prob_list]


def _build_huffman_code_lengths(probs: List[float], token_ids: List[int]) -> Dict[int, int]:
    n = len(token_ids)
    if n == 0:
        return {}
    if n == 1:
        return {token_ids[0]: 1}

    heap: List[Tuple[float, int, _HuffNode]] = []
    for p, tid in zip(probs, token_ids):
        node = _HuffNode(p=p, token_id=tid)
        heapq.heappush(heap, (node.p, tid, node))

    while len(heap) > 1:
        p1, t1, n1 = heapq.heappop(heap)
        p2, t2, n2 = heapq.heappop(heap)

        if (t2, p2) < (t1, p1):
            (p1, t1, n1), (p2, t2, n2) = (p2, t2, n2), (p1, t1, n1)

        merged = _HuffNode(p=p1 + p2, token_id=None, left=n1, right=n2)
        tie = min(t1, t2)
        heapq.heappush(heap, (merged.p, tie, merged))

    root = heap[0][2]

    lengths: Dict[int, int] = {}
    stack: List[Tuple[_HuffNode, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if node.token_id is not None:
            lengths[node.token_id] = max(1, depth)
            continue
        if node.right is not None:
            stack.append((node.right, depth + 1))
        if node.left is not None:
            stack.append((node.left, depth + 1))
    return lengths


def _canonical_huffman_codes(code_lengths: Dict[int, int]) -> Dict[int, str]:
    if not code_lengths:
        return {}

    items = sorted(code_lengths.items(), key=lambda x: (x[1], x[0]))

    codes: Dict[int, str] = {}
    code = 0
    prev_len = items[0][1]
    for tid, L in items:
        if L > prev_len:
            code <<= (L - prev_len)
            prev_len = L
        codes[tid] = format(code, f'0{L}b')
        code += 1
    return codes


def build_huffman_codes(prob_list: List[float], indices_list: List[int]) -> Dict[int, str]:
    probs = _normalize_probs(list(prob_list))
    token_ids = list(indices_list)

    pairs = sorted(zip(probs, token_ids), key=lambda x: (-x[0], x[1]))
    probs = [p for p, _ in pairs]
    token_ids = [tid for _, tid in pairs]

    lengths = _build_huffman_code_lengths(probs, token_ids)
    return _canonical_huffman_codes(lengths)


# ==================== 兼容旧接口：adg_grouping (保留但不再使用) ====================

def adg_grouping(prob: List[float], indices: List[int], num_groups: int) -> List[Tuple[List[float], List[int]]]:
    if num_groups <= 0:
        return []
    prob = list(prob)
    indices = list(indices)
    total = sum(prob)
    if total > 0:
        prob = [p / total for p in prob]

    groups: List[Tuple[List[float], List[int]]] = [([], []) for _ in range(num_groups)]
    for i, (p, tid) in enumerate(zip(prob, indices)):
        g = i % num_groups
        groups[g][0].append(p)
        groups[g][1].append(tid)
    return groups


# ==================== Huffman 隐写：编码/解码（保持旧函数名） ====================

def _highest_prob_token(prob_list: List[float], indices_list: List[int]) -> int:
    probs = _normalize_probs(list(prob_list))
    pairs = sorted(zip(probs, list(indices_list)), key=lambda x: (-x[0], x[1]))
    return pairs[0][1]


def adg_encode(
    prob_list: List[float],
    indices_list: List[int],
    secret_bits: str,
    current_bit_index: int,
    debug: bool = False,
) -> Tuple[int, int]:
    """编码：用霍夫曼码把秘密比特映射为一个 token。

    返回:
      chosen_token_id, payload_bits_used

    注意：当 secret 已经到末尾但当前前缀尚未形成完整码字时，
    会选择一个“码字以当前剩余 bit 为前缀”的 token，
    但 bits_used 仍然只记真实 payload 位数。
    这样解码端用 remaining_bits 截断即可恢复原 payload，
    不会再出现尾部污染。
    """
    if not indices_list:
        return -1, 0

    codes = build_huffman_codes(prob_list, indices_list)
    code_to_token = {code: tid for tid, code in codes.items()}

    prefixes = {''}
    for c in code_to_token:
        for k in range(1, len(c) + 1):
            prefixes.add(c[:k])

    if current_bit_index >= len(secret_bits):
        return _highest_prob_token(prob_list, indices_list), 0

    prefix = ''
    i = current_bit_index
    max_len = max((len(c) for c in code_to_token), default=1)

    while i < len(secret_bits) and len(prefix) < max_len:
        b = secret_bits[i]
        if b not in ('0', '1'):
            raise ValueError(f"secret_bits 只能包含 '0'/'1'，但在位置 {i} 得到: {b!r}")
        prefix += b
        i += 1

        if prefix in code_to_token:
            chosen = code_to_token[prefix]
            used = len(prefix)
            if debug:
                print(f"    [HUF-E] 选择token {chosen}, 嵌入{used}bits, code={prefix}")
            return chosen, used

        if prefix not in prefixes:
            break

    # secret 已读完，但当前前缀仍不是完整码字：补到某个合法码字上。
    if i >= len(secret_bits) and prefix:
        matches = [
            (len(code), code, tid)
            for tid, code in codes.items()
            if code.startswith(prefix)
        ]
        if matches:
            _, matched_code, chosen = min(matches)
            used = len(prefix)
            if debug:
                print(
                    f"    [HUF-E] 尾部补齐选择token {chosen}, payload_bits={used}, "
                    f"code={matched_code}, payload_prefix={prefix}"
                )
            return chosen, used

    fallback = _highest_prob_token(prob_list, indices_list)
    if debug:
        print(f"    [HUF-E] 无法匹配合法码字，回退选择token {fallback}")
    return fallback, 0


def adg_decode(
    prob_list: List[float],
    indices_list: List[int],
    actual_token_id: int,
    remaining_bits: int,
    debug: bool = False,
) -> str:
    if remaining_bits <= 0 or not indices_list:
        return ''

    codes = build_huffman_codes(prob_list, indices_list)
    code = codes.get(actual_token_id, '')

    if not code:
        if debug:
            print(f"    [HUF-D] token {actual_token_id} 不在候选集中，提取0bits")
        return ''

    out = code[:remaining_bits]
    if debug:
        print(f"    [HUF-D] token {actual_token_id}, 提取{len(out)}bits, code={code}")
    return out
