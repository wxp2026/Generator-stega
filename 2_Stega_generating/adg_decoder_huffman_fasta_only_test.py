
#!/usr/bin/env python3
"""
Huffman DNA隐写术解码器 - FASTA鲁棒性测试版

用途：
1. 强制仅根据当前 FASTA 文件重新分词并解码；
2. 完全禁用 token_ids_*.json 与 metadata 中 token_ids_file 的影响；
3. 用于测试对 .fasta DNA 序列做扰动/删除/替换后的解码鲁棒性。

和原版相比的关键变化：
- 删除了 exact_token_ids 模式，只保留 retokenized_fasta_only 模式。
- 不再自动搜索 token_ids_*.json，也不读取 metadata 里的 token_ids_file。
- 模型上下文严格来自当前 FASTA 的前缀序列，而不是历史 token 轨迹。
- 增加了若干一致性输出，便于观察“修改 10% 后 token 数是否下降、实际提取到多少 bit”。

注意：
- 该脚本仍可使用 metadata 中的 prompt 长度、temperature、top_k、reading_frame、
  embedded_bits/payload_bits、原始 secret 等参数。
- 这里“根据 FASTA 文件解码”指：token 划分、上下文构造、实际 token 都来自当前 FASTA。
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from adg_core_huffman_fixed import adg_decode

STOP_CODONS = ['TAA', 'TAG', 'TGA']
START_CODON = 'ATG'

P450_CONSERVED_MOTIFS = {
    "cys_heme_signature": {"pattern": r"F[A-Z]{2}G[A-Z]{3}C[A-Z]G", "padding": 8},
    "helix_C_WxxxR": {"pattern": r"W[A-Z]{3}R", "padding": 6},
    "helix_I_oxygen_binding": {"pattern": r"[AG][AG][A-Z][DE]T[TS]", "padding": 6},
    "acid_alcohol_pair_I_helix": {"pattern": r"[DE][A-Z]{2}[TS]", "padding": 6},
    "helix_K_ExxR": {"pattern": r"E[A-Z]{2}R", "padding": 6},
    "PERF_motif": {"pattern": r"P[ED]RF", "padding": 6},
    "meander_region": {"pattern": r"[A-Z]{3,6}C[A-Z]{2,4}G", "padding": 6},
}


def filter_candidates(
    probs: torch.Tensor,
    indices: torch.Tensor,
    current_seq: str,
    reading_frame: int,
    stop_token_ids: List[int],
    start_token_ids: List[int],
    tokenizer,
    current_position: int,
) -> Tuple[List[float], List[int]]:
    filtered_probs: List[float] = []
    filtered_indices: List[int] = []

    for idx in range(len(probs)):
        token_id = indices[idx].item()
        prob = probs[idx].item()

        if token_id in stop_token_ids or token_id in start_token_ids:
            continue

        try:
            token_dna = tokenizer.decode([token_id], skip_special_tokens=True)

            offset_in_frame = (current_position - reading_frame) % 3
            first_codon_start = 0 if offset_in_frame == 0 else (3 - offset_in_frame)

            contains_stop = False
            if first_codon_start + 3 <= len(token_dna):
                for i in range(first_codon_start, len(token_dna) - 2, 3):
                    codon = token_dna[i:i + 3]
                    if codon in STOP_CODONS:
                        contains_stop = True
                        break
            if contains_stop:
                continue

            current_pos = len(current_seq)
            boundary_offset = (current_pos - reading_frame) % 3

            if boundary_offset == 0:
                if len(token_dna) >= 3 and token_dna[0:3] in STOP_CODONS:
                    continue
            elif boundary_offset == 1:
                if len(current_seq) >= 2 and len(token_dna) >= 1:
                    if current_seq[-2:] + token_dna[0] in STOP_CODONS:
                        continue
            elif boundary_offset == 2:
                if len(current_seq) >= 1 and len(token_dna) >= 2:
                    if current_seq[-1:] + token_dna[0:2] in STOP_CODONS:
                        continue

            filtered_probs.append(prob)
            filtered_indices.append(token_id)
        except Exception:
            continue

    if len(filtered_indices) == 0:
        for idx in range(len(probs)):
            token_id = indices[idx].item()
            prob = probs[idx].item()
            if token_id not in stop_token_ids and token_id not in start_token_ids:
                filtered_probs.append(prob)
                filtered_indices.append(token_id)

    return filtered_probs, filtered_indices


def load_reference_sequence(parquet_file: str, protein_id: str) -> Tuple[str, str]:
    df = pd.read_parquet(parquet_file)
    target = df[df['protein_id'] == protein_id]
    if target.empty:
        raise ValueError(f"未找到蛋白质ID: {protein_id}")
    return target.iloc[0]['dna_sequence'], target.iloc[0]['protein_sequence']


class ConservedDomainMapper:
    def __init__(self, reference_dna: str, reference_protein: str, reading_frame: int = 0):
        self.reference_dna = reference_dna
        self.reference_protein = reference_protein
        self.reading_frame = reading_frame
        self.conserved_regions: List[Dict] = []

    def map_conserved_domains(self, prompt_length: int = 0) -> List[Dict]:
        self.conserved_regions = []
        temp_regions = []

        for motif_name, info in P450_CONSERVED_MOTIFS.items():
            for match in re.finditer(info['pattern'], self.reference_protein):
                aa_start, aa_end = match.start(), match.end()
                dna_start = self.reading_frame + aa_start * 3
                dna_end = self.reading_frame + aa_end * 3
                padding = info['padding'] * 3

                temp_regions.append({
                    'start': max(0, dna_start - padding),
                    'end': min(len(self.reference_dna), dna_end + padding),
                    'name': motif_name,
                })

        temp_regions.sort(key=lambda x: x['start'])
        merged = []
        for region in temp_regions:
            if not merged or region['start'] > merged[-1]['end']:
                merged.append(region)
            else:
                merged[-1]['end'] = max(merged[-1]['end'], region['end'])

        for region in merged:
            start = (region['start'] // 6) * 6
            end = region['end']
            if end % 6 != 0:
                end = ((end // 6) + 1) * 6
            if end > prompt_length:
                self.conserved_regions.append({
                    'start': max(start, prompt_length),
                    'end': end,
                    'name': region['name'],
                })
        return self.conserved_regions

    def get_conserved_positions(self) -> Set[int]:
        positions: Set[int] = set()
        for region in self.conserved_regions:
            positions.update(range(region['start'], region['end']))
        return positions


def load_metadata(meta_file: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                metadata[key] = value
    return metadata


def load_sequence(fasta_file: str) -> str:
    sequence = []
    with open(fasta_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith('>'):
                sequence.append(line.strip())
    return ''.join(sequence)


def build_token_map_from_sequence(sequence: str, tokenizer) -> Tuple[List[int], List[Tuple[int, int, int, str]]]:
    token_ids = tokenizer.encode(sequence, add_special_tokens=False)
    token_map: List[Tuple[int, int, int, str]] = []
    pos = 0
    for tid in token_ids:
        text = tokenizer.decode([tid], skip_special_tokens=True)
        start = pos
        end = pos + len(text)
        token_map.append((start, end, tid, text))
        pos = end
    return token_ids, token_map


def extract_hidden_bits_fasta_only(
    full_sequence: str,
    model,
    tokenizer,
    prompt_length: int,
    target_bits: int,
    temperature: float,
    top_k: int,
    reading_frame: int,
    conserved_positions: Set[int],
    min_prob_threshold: float = 1e-10,
    debug: bool = False,
) -> Tuple[str, Dict]:
    device = next(model.parameters()).device

    stop_ids: List[int] = []
    for codon in STOP_CODONS:
        stop_ids.extend(tokenizer.encode(codon, add_special_tokens=False))
    start_ids = tokenizer.encode(START_CODON, add_special_tokens=False)

    extracted: List[str] = []
    stats = {
        'processed': 0,
        'skipped': 0,
        'conserved': 0,
        'special_empty': 0,
        'token_count_total': 0,
        'token_count_after_prompt': 0,
        'prompt_token_count': 0,
        'bp_covered_by_tokens': 0,
    }

    def total_bits() -> int:
        return sum(len(b) for b in extracted)

    def in_conserved(pos: int, length: int) -> bool:
        return any(p in conserved_positions for p in range(pos, pos + max(length, 1)))

    all_token_ids, token_map = build_token_map_from_sequence(full_sequence, tokenizer)
    prompt_token_count = len(tokenizer.encode(full_sequence[:prompt_length], add_special_tokens=False))

    stats['token_count_total'] = len(all_token_ids)
    stats['prompt_token_count'] = prompt_token_count
    stats['token_count_after_prompt'] = max(0, len(all_token_ids) - prompt_token_count)
    stats['bp_covered_by_tokens'] = token_map[-1][1] if token_map else 0

    if debug:
        print(f"\n[提取] FASTA-only 模式")
        print(f"[提取] 序列长度={len(full_sequence)} bp, token总数={len(all_token_ids)}")
        print(f"[提取] prompt_bp={prompt_length}, prompt_token_count={prompt_token_count}")
        print(f"[提取] 目标bits={target_bits}, 温度={temperature}, top_k={top_k if top_k > 0 else '禁用'}")

    with torch.inference_mode():
        for tok_idx in range(prompt_token_count, len(token_map)):
            if total_bits() >= target_bits:
                break

            start, end, actual_id, text = token_map[tok_idx]

            if text == '':
                stats['special_empty'] += 1
                continue

            if in_conserved(start, end - start):
                stats['conserved'] += 1
                if debug:
                    print(f"  Token {tok_idx} @{start}: 跳过保守域")
                continue

            context = full_sequence[:start]
            input_ids = tokenizer.encode(context, add_special_tokens=False, return_tensors='pt').to(device)

            logits = model(input_ids).logits[0, -1, :]
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=0)

            if top_k > 0:
                top_k_probs, top_k_indices = torch.topk(probs, min(top_k, len(probs)))
                top_k_probs = top_k_probs / top_k_probs.sum()
            else:
                valid_mask = probs >= min_prob_threshold
                valid_indices = torch.where(valid_mask)[0]
                valid_probs = probs[valid_indices]
                if len(valid_indices) == 0:
                    stats['skipped'] += 1
                    continue
                sorted_probs, sorted_local_indices = torch.sort(valid_probs, descending=True)
                top_k_indices = valid_indices[sorted_local_indices]
                top_k_probs = sorted_probs

            if len(top_k_probs) == 0:
                stats['skipped'] += 1
                continue

            if len(top_k_probs) > 1 and top_k_probs[0] < top_k_probs[1]:
                sorted_probs, sorted_local_indices = torch.sort(top_k_probs, descending=True)
                top_k_indices = top_k_indices[sorted_local_indices]
                top_k_probs = sorted_probs

            filt_probs, filt_indices = filter_candidates(
                top_k_probs,
                top_k_indices,
                context,
                reading_frame,
                stop_ids,
                start_ids,
                tokenizer,
                start,
            )

            if not filt_probs:
                stats['skipped'] += 1
                continue

            if actual_id not in filt_indices:
                stats['skipped'] += 1
                if debug:
                    print(f"  Token {tok_idx} @{start}: '{text}' (id={actual_id}) 不在候选列表中")
                continue

            remaining_bits = target_bits - total_bits()
            bits_str = adg_decode(filt_probs, filt_indices, actual_id, remaining_bits, debug)

            if bits_str:
                bits_str = bits_str[:remaining_bits]
                extracted.append(bits_str)
                stats['processed'] += 1
                if debug:
                    print(f"  Token {tok_idx} @{start}: '{text}' -> 提取 {bits_str} (累计 {total_bits()}/{target_bits})")
            else:
                stats['skipped'] += 1

    result = ''.join(extracted)
    stats['total'] = len(result)
    stats['mode'] = 'retokenized_fasta_only'
    return result, stats


def main():
    parser = argparse.ArgumentParser(description="Huffman DNA隐写术解码器 (FASTA鲁棒性测试版)")
    parser.add_argument("--meta", default=r".\huffman\outputs\20260506_134827\metadata_20260506_134827.txt", help="元数据文件")
    parser.add_argument("--fasta", default=r".\huffman\outputs\20260506_134827\sequence_20260506_134827_swap.fasta", help="FASTA文件")
    parser.add_argument("--model", default=r"C:\huggingface\checkpoint-human", help="模型路径")
    parser.add_argument("--output", default="./decode_results_fasta_only", help="输出目录")
    parser.add_argument("--no-compare", action="store_true", help="不与 metadata 中原始 secret 比较")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    print("=" * 70)
    print("Huffman DNA隐写术解码器 (FASTA鲁棒性测试版)")
    print("=" * 70)

    print("\n[1] 加载元数据...")
    meta = load_metadata(args.meta)
    prompt_len = int(meta.get('truncated_prompt_length', 0))
    actual_prompt = meta.get('actual_prompt', '')
    temperature = float(meta.get('temperature', 1.0))
    top_k = int(meta.get('top_k', 0))
    reading_frame = int(meta.get('reading_frame', 0))
    original_secret = meta.get('secret', '')

    if 'embedded_bits' in meta:
        target_bits = int(meta['embedded_bits'])
    elif 'payload_bits' in meta:
        target_bits = int(meta['payload_bits'])
    else:
        target_bits = len(original_secret)

    ref_parquet = meta.get('reference_parquet', '')
    ref_protein_id = meta.get('reference_protein_id', '')

    print(f"    prompt长度(bp): {prompt_len}")
    print(f"    温度: {temperature}")
    print(f"    top_k: {top_k if top_k > 0 else '禁用'}")
    print(f"    目标bits: {target_bits}")
    print(f"    token_ids: 已强制禁用（无论 metadata 或同目录是否存在）")

    print("\n[2] 加载保守域...")
    conserved: Set[int] = set()
    if ref_parquet and ref_protein_id and os.path.exists(ref_parquet):
        try:
            dna, protein = load_reference_sequence(ref_parquet, ref_protein_id)
            mapper = ConservedDomainMapper(dna, protein, reading_frame)
            mapper.map_conserved_domains(prompt_len)
            conserved = mapper.get_conserved_positions()
            print(f"    保守域位置数: {len(conserved)}")
        except Exception as e:
            print(f"    警告: {e}")
    else:
        print("    未使用保守域（缺少 reference_parquet/reference_protein_id 或文件不存在）")

    print("\n[3] 加载 FASTA 序列...")
    sequence = load_sequence(args.fasta)
    print(f"    FASTA长度: {len(sequence)} bp")

    if actual_prompt and sequence.startswith(actual_prompt) and len(actual_prompt) != prompt_len:
        print(f"    校正 prompt 长度: {prompt_len} -> {len(actual_prompt)}")
        prompt_len = len(actual_prompt)

    print("\n[4] 加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    设备: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    token_count_now = len(tokenizer.encode(sequence, add_special_tokens=False))
    prompt_token_count_now = len(tokenizer.encode(sequence[:prompt_len], add_special_tokens=False))
    print("\n[5] 当前 FASTA 重新分词统计...")
    print(f"    当前token总数: {token_count_now}")
    print(f"    当前prompt token数: {prompt_token_count_now}")
    print(f"    prompt后token数: {max(0, token_count_now - prompt_token_count_now)}")

    print("\n[6] 开始提取（仅按当前 FASTA 解码）...")
    result, stats = extract_hidden_bits_fasta_only(
        full_sequence=sequence,
        model=model,
        tokenizer=tokenizer,
        prompt_length=prompt_len,
        target_bits=target_bits,
        temperature=temperature,
        top_k=top_k,
        reading_frame=reading_frame,
        conserved_positions=conserved,
        debug=args.debug,
    )

    print("\n" + "=" * 70)
    print("[结果]")
    print(f"  提取: {len(result)}/{target_bits} bits")
    print(f"  模式: {stats['mode']}")
    print(f"  当前FASTA token总数: {stats['token_count_total']}")
    print(f"  当前FASTA prompt token数: {stats['prompt_token_count']}")
    print(f"  当前FASTA prompt后token数: {stats['token_count_after_prompt']}")
    print(f"  token覆盖bp: {stats['bp_covered_by_tokens']}/{len(sequence)}")
    print(f"  处理tokens: {stats['processed']}")
    print(f"  跳过: {stats['skipped']}")
    print(f"  保守域: {stats['conserved']}")
    print(f"  空special token: {stats['special_empty']}")

    compare_secret = original_secret[:target_bits] if (original_secret and not args.no_compare) else ''
    acc = None
    matches = None
    min_len = min(len(compare_secret), len(result)) if compare_secret else 0

    if compare_secret:
        matches = sum(1 for i in range(min_len) if compare_secret[i] == result[i])
        acc = (matches / min_len * 100) if min_len > 0 else 0.0
        print(f"\n  对原始secret准确率: {acc:.2f}% ({matches}/{min_len})")
        print(f"\n  原始(前80): {compare_secret[:80]}")
        print(f"  提取(前80): {result[:80]}")

        if compare_secret == result:
            print("\n  ✓ 与 metadata 中原始secret完全匹配")
        else:
            for i in range(min_len):
                if compare_secret[i] != result[i]:
                    s = max(0, i - 5)
                    print(f"\n  首个不匹配: 位置 {i}")
                    print(f"    原始[{s}:{i + 10}]: {compare_secret[s:i + 10]}")
                    print(f"    提取[{s}:{i + 10}]: {result[s:i + 10]}")
                    break
    else:
        print("\n  未进行与原始secret的比较")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'extraction_result_fasta_only.txt'
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(f"mode: {stats['mode']}\n")
        f.write(f"fasta_file: {args.fasta}\n")
        f.write(f"fasta_length_bp: {len(sequence)}\n")
        f.write(f"token_count_total: {stats['token_count_total']}\n")
        f.write(f"prompt_token_count: {stats['prompt_token_count']}\n")
        f.write(f"token_count_after_prompt: {stats['token_count_after_prompt']}\n")
        f.write(f"bp_covered_by_tokens: {stats['bp_covered_by_tokens']}\n")
        f.write(f"target_bits: {target_bits}\n")
        f.write(f"extracted_bits: {len(result)}\n")
        f.write(f"processed_tokens: {stats['processed']}\n")
        f.write(f"skipped_tokens: {stats['skipped']}\n")
        f.write(f"conserved_tokens: {stats['conserved']}\n")
        f.write(f"special_empty_tokens: {stats['special_empty']}\n")
        if compare_secret:
            f.write(f"accuracy_vs_original_secret: {acc:.2f}\n")
            f.write(f"original_secret: {compare_secret}\n")
        f.write(f"decoded_bits: {result}\n")
    print(f"\n  保存到: {out_file}")


if __name__ == '__main__':
    main()
