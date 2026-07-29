
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.stopping_criteria import StoppingCriteria
import math
import re
from typing import List, Dict, Tuple, Optional, Set
import argparse
import random
import numpy as np
import json
import os
from pathlib import Path
import pandas as pd

# 导入ADG核心模块
from adg_core_huffman_fixed import bits2int, int2bits, near, adg_grouping, adg_encode, adg_decode

STOP_CODONS = ['TAA', 'TAG', 'TGA']
START_CODON = 'ATG'

DEFAULT_PROMOTER = "ATGCTTGTCACAGCGGGTTCCCTACTAGGGGCCATTTGGACC"

# P450保守域定义
P450_CONSERVED_MOTIFS = {
    "cys_heme_signature": {
        "pattern": r"F[A-Z]{2}G[A-Z]{3}C[A-Z]G",
        "description": "Cys血红素配体签名域",
        "padding": 8,
    },
    "helix_C_WxxxR": {
        "pattern": r"W[A-Z]{3}R",
        "description": "C螺旋保守WxxxR模体",
        "padding": 6,
    },
    "helix_I_oxygen_binding": {
        "pattern": r"[AG][AG][A-Z][DE]T[TS]",
        "description": "I螺旋氧结合模体",
        "padding": 6,
    },
    "acid_alcohol_pair_I_helix": {
        "pattern": r"[DE][A-Z]{2}[TS]",
        "description": "I螺旋acid-alcohol pair",
        "padding": 6,
    },
    "helix_K_ExxR": {
        "pattern": r"E[A-Z]{2}R",
        "description": "K螺旋保守ExxR模体",
        "padding": 6,
    },
    "PERF_motif": {
        "pattern": r"P[ED]RF",
        "description": "β1-4区PERF模体",
        "padding": 6,
    },
    "meander_region": {
        "pattern": r"[A-Z]{3,6}C[A-Z]{2,4}G",
        "description": "Meander区段",
        "padding": 6,
    },
}


# ==================== Token过滤函数 (编码器/解码器共用) ====================

def filter_candidates(
        probs: torch.Tensor,
        indices: torch.Tensor,
        current_seq: str,
        reading_frame: int,
        stop_token_ids: List[int],
        start_token_ids: List[int],
        tokenizer,
        current_position: int
) -> Tuple[List[float], List[int]]:

    filtered_probs = []
    filtered_indices = []

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
    """从parquet文件加载参考序列"""
    df = pd.read_parquet(parquet_file)
    target = df[df['protein_id'] == protein_id]

    if target.empty:
        raise ValueError(f"未找到蛋白质ID: {protein_id}")

    return target.iloc[0]['dna_sequence'], target.iloc[0]['protein_sequence']


class ConservedDomainMapper:
    """保守域映射器"""

    def __init__(self, reference_dna: str, reference_protein: str,
                 reading_frame: int = 0, debug: bool = False):
        self.reference_dna = reference_dna
        self.reference_protein = reference_protein
        self.reading_frame = reading_frame
        self.debug = debug
        self.conserved_regions = []
        self.conserved_dna_segments = {}

    def map_conserved_domains(self, prompt_length: int = 0) -> List[Dict]:
        """在参考序列中识别保守域"""
        self.conserved_regions = []
        self.conserved_dna_segments = {}

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
                    'name': motif_name
                })

        # 合并重叠区域
        temp_regions.sort(key=lambda x: x['start'])
        merged = []
        for region in temp_regions:
            if not merged or region['start'] > merged[-1]['end']:
                merged.append(region)
            else:
                merged[-1]['end'] = max(merged[-1]['end'], region['end'])
                merged[-1]['name'] = f"{merged[-1]['name']},{region['name']}"

        # 对齐到6bp边界
        for region in merged:
            start = (region['start'] // 6) * 6
            end = region['end']
            if end % 6 != 0:
                end = ((end // 6) + 1) * 6

            if end > prompt_length:
                actual_start = max(start, prompt_length)
                dna_segment = self.reference_dna[actual_start:end]

                self.conserved_regions.append({
                    'start': actual_start,
                    'end': end,
                    'name': region['name'],
                    'dna_segment': dna_segment
                })
                self.conserved_dna_segments[actual_start] = dna_segment

        return self.conserved_regions

    def get_conserved_positions(self) -> Set[int]:

        positions = set()
        for region in self.conserved_regions:
            positions.update(range(region['start'], region['end']))
        return positions

    def get_dna_segment(self, position: int) -> Optional[str]:

        for region in self.conserved_regions:
            if region['start'] <= position < region['end']:
                offset = position - region['start']
                return region['dna_segment'][offset:]
        return None


class ADGEncoder(LogitsProcessor):


    def __init__(
            self,
            secret_bits: str,
            tokenizer,
            prompt_length: int,
            conserved_mapper: ConservedDomainMapper,
            reading_frame: int = 0,
            target_length: int = 1500,
            temperature: float = 1.0,
            top_k: int = 10,  # 默认使用top_k=10
            min_prob_threshold: float = 1e-10,
            debug: bool = False,
            max_prob_output: int = 30  # 最多输出前30个选词的概率
    ):
        self.secret_bits = secret_bits
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.conserved_mapper = conserved_mapper
        self.conserved_positions = conserved_mapper.get_conserved_positions()
        self.reading_frame = reading_frame
        self.target_length = target_length
        self.temperature = temperature
        self.top_k = top_k
        self.min_prob_threshold = min_prob_threshold
        self.debug = debug
        self.max_prob_output = max_prob_output

        self.current_bit_index = 0
        self.current_position = prompt_length
        self.tokens_embedded = 0
        self.tokens_in_conserved = 0
        self.tokens_skipped = 0

        # 预编码禁止的tokens
        self.stop_codon_token_ids = []
        for codon in STOP_CODONS:
            self.stop_codon_token_ids.extend(
                tokenizer.encode(codon, add_special_tokens=False)
            )
        self.start_codon_token_ids = tokenizer.encode(START_CODON, add_special_tokens=False)

        if self.debug:
            print(f"\n[ADG编码器初始化]")
            print(f"  秘密长度: {len(secret_bits)} bits")
            print(f"  Prompt长度: {prompt_length} bp")
            print(f"  温度: {temperature}")
            print(f"  Top-k: {top_k if top_k > 0 else '禁用'}")
            print(f"  保守域区域数: {len(conserved_mapper.conserved_regions)}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:



        current_seq = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        self.current_position = len(current_seq)

        # 检查保守域
        dna_segment = self.conserved_mapper.get_dna_segment(self.current_position)
        if dna_segment is not None:
            next_token_dna = dna_segment[:6] if len(dna_segment) >= 6 else dna_segment
            real_token_ids = self.tokenizer.encode(next_token_dna, add_special_tokens=False)

            if len(real_token_ids) > 0:
                new_scores = torch.full_like(scores[0], -float('inf'))
                new_scores[real_token_ids[0]] = 100.0
                scores[0] = new_scores
                self.tokens_in_conserved += 1

                if self.debug:
                    print(f"  位置 {self.current_position}: 保守域 -> '{next_token_dna}'")
                return scores

        # 检查是否已嵌入完成
        if self.current_bit_index >= len(self.secret_bits):
            # 进入 greedy 模式：只允许选择当前概率最大的 token
            greedy_token = torch.argmax(scores, dim=-1)  # shape: [batch]
            new_scores = torch.full_like(scores, float('-inf'))
            new_scores.scatter_(1, greedy_token.unsqueeze(1), 0.0)
            return new_scores

        # ===== 步骤1: 应用温度 =====
        scaled_scores = scores[0] / self.temperature

        # ===== 步骤2: 计算概率 =====
        probs = F.softmax(scaled_scores, dim=-1)

        # ===== 步骤3: 应用top_k裁剪 =====
        if self.top_k > 0:
            # 只保留top_k个最高概率的token
            top_k_probs, top_k_indices = torch.topk(probs, min(self.top_k, len(probs)))
            # 重新归一化
            top_k_probs = top_k_probs / top_k_probs.sum()
        else:
            # 不使用top_k，使用所有高于阈值的token
            valid_mask = probs >= self.min_prob_threshold
            valid_indices = torch.where(valid_mask)[0]
            valid_probs = probs[valid_indices]

            if len(valid_indices) == 0:
                return scores

            # 排序（降序）
            sorted_probs, sorted_local_indices = torch.sort(valid_probs, descending=True)
            top_k_indices = valid_indices[sorted_local_indices]
            top_k_probs = sorted_probs

        if len(top_k_probs) == 0:
            return scores

        # 确保是降序的
        if len(top_k_probs) > 1 and top_k_probs[0] < top_k_probs[1]:
            sorted_probs, sorted_local_indices = torch.sort(top_k_probs, descending=True)
            top_k_indices = top_k_indices[sorted_local_indices]
            top_k_probs = sorted_probs

        # ===== 步骤4: 过滤候选token =====
        filtered_probs, filtered_indices = filter_candidates(
            top_k_probs, top_k_indices, current_seq, self.reading_frame,
            self.stop_codon_token_ids, self.start_codon_token_ids,
            self.tokenizer, self.current_position
        )

        if len(filtered_probs) == 0:
            return scores

        # ===== 步骤5: ADG编码 =====
        chosen_token_id, bits_embedded = adg_encode(
            filtered_probs, filtered_indices,
            self.secret_bits, self.current_bit_index,
            debug=self.debug
        )

        if chosen_token_id < 0:
            return scores

        # 强制选择该token
        new_scores = torch.full_like(scores[0], -float('inf'))
        new_scores[chosen_token_id] = 100.0
        scores[0] = new_scores

        self.current_bit_index += bits_embedded
        self.tokens_embedded += 1

        # ===== 输出选词概率信息（仅前max_prob_output个） =====
        if self.tokens_embedded <= self.max_prob_output:
            # 计算归一化概率
            filtered_probs_sum = sum(filtered_probs)
            normalized_probs = [p / filtered_probs_sum for p in filtered_probs]

            # 找到选中token的概率
            chosen_idx = filtered_indices.index(chosen_token_id) if chosen_token_id in filtered_indices else -1
            chosen_prob = normalized_probs[chosen_idx] if chosen_idx >= 0 else 0.0
            chosen_token_text = self.tokenizer.decode([chosen_token_id], skip_special_tokens=True)

            print(f"\n{'=' * 60}")
            print(f"[选词 #{self.tokens_embedded}] 位置: {self.current_position} bp")
            print(f"{'=' * 60}")
            print(f"候选token数量: {len(filtered_indices)}")
            print(f"嵌入bits数: {bits_embedded}")
            print(f"累计嵌入: {self.current_bit_index}/{len(self.secret_bits)} bits")
            print(f"\n--- 候选token概率分布 (Top {len(filtered_indices)}) ---")

            # 显示所有候选token及其概率
            for i in range(len(filtered_indices)):
                token_id = filtered_indices[i]
                prob = normalized_probs[i]
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                marker = " <-- 选中" if token_id == chosen_token_id else ""
                print(
                    f"  [{i + 1}] Token: '{token_text}' (ID: {token_id}) | 概率: {prob:.6f} ({prob * 100:.4f}%){marker}")

            print(f"\n>>> 最终选择: '{chosen_token_text}' | 概率: {chosen_prob:.6f} ({chosen_prob * 100:.4f}%)")
            print(f"{'=' * 60}\n")

        if self.debug:
            token_text = self.tokenizer.decode([chosen_token_id], skip_special_tokens=True)
            print(f"  [DEBUG] 位置 {self.current_position}: ADG嵌入 {bits_embedded} bits "
                  f"(total: {self.current_bit_index}/{len(self.secret_bits)}) -> '{token_text}'")

        return scores


class LengthStoppingCriteria(StoppingCriteria):
    """长度停止标准"""

    def __init__(self, encoder, tokenizer, target_length: int):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.target_length = target_length
        self.stop_reason = "unknown"

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        current_seq = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        current_length = len(current_seq)

        if current_length >= self.target_length:
            self.stop_reason = "target_length"
            return True

        return False


def generate_with_adg(
        model_path: str,
        secret_message: str,
        output_dir: str,
        reference_parquet: Optional[str] = None,
        reference_protein_id: Optional[str] = None,
        target_length: int = 1500,
        temperature: float = 1.0,
        top_k: int = 10,
        promoter: Optional[str] = None,
        reading_frame: int = 0,
        debug: bool = False
):
    """
    使用Huffman隐写算法生成隐写序列

    参数:
        temperature: ADG采样温度，只在LogitsProcessor中应用一次
        top_k: top-k采样参数，默认10
    """

    print("=" * 70)
    print("Huffman DNA隐写术编码器 ")
    print("=" * 70)

    # 加载模型
    print("\n[1] 加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    # 准备保守域映射
    print("\n[2] 准备保守域映射...")
    if reference_parquet and os.path.exists(reference_parquet):
        ref_dna, ref_protein = load_reference_sequence(reference_parquet, reference_protein_id)
        print(f"  参考序列: {reference_protein_id}")
        print(f"  DNA长度: {len(ref_dna)} bp")
    else:
        ref_dna = ""
        ref_protein = ""
        print("  未使用参考序列")

    # 准备prompt
    if promoter:
        prompt = promoter
    elif ref_dna:
        prompt = ref_dna[:48]
    else:
        prompt = DEFAULT_PROMOTER

    # 对齐prompt长度到6bp边界
    prompt_length = (len(prompt) // 6) * 6
    prompt = prompt[:prompt_length]

    print(f"\n[3] Prompt: {prompt[:50]}...")
    print(f"  长度: {prompt_length} bp")

    # 创建保守域映射器
    mapper = ConservedDomainMapper(ref_dna, ref_protein, reading_frame, debug)
    conserved_regions = mapper.map_conserved_domains(prompt_length)

    print(f"\n[4] 保守域: {len(conserved_regions)} 个区域")

    # 创建编码器
    encoder = ADGEncoder(
        secret_bits=secret_message,
        tokenizer=tokenizer,
        prompt_length=prompt_length,
        conserved_mapper=mapper,
        reading_frame=reading_frame,
        target_length=target_length,
        temperature=temperature,
        top_k=top_k,
        debug=debug
    )

    # 创建停止条件
    stopping_criteria = LengthStoppingCriteria(encoder, tokenizer, target_length)

    # 生成
    print(f"\n[5] 开始生成...")
    print(f"  秘密长度: {len(secret_message)} bits")
    print(f"  目标长度: {target_length} bp")
    print(f"  温度: {temperature}")
    print(f"  Top-k: {top_k if top_k > 0 else '禁用'}")

    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    prompt_token_count = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)  # 全1，表示所有token都需要attention

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,  # 显式设置attention_mask
            max_length=target_length // 3 + len(input_ids[0]),
            do_sample=True,
            temperature=1.0,  # 重要：设为1.0，温度在LogitsProcessor中应用
            top_k=0,  # 重要：禁用generate的top_k，在LogitsProcessor中处理
            top_p=1.0,  # 禁用top_p
            logits_processor=[encoder],
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id
        )

    final_sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 截断到目标长度
    if len(final_sequence) > target_length:
        final_length = (target_length // 3) * 3
        final_sequence = final_sequence[:final_length]

    print(f"\n[6] 生成完成")
    print(f"  最终长度: {len(final_sequence)} bp")
    print(f"  嵌入bits: {encoder.current_bit_index}/{len(secret_message)}")
    print(f"  ADG tokens: {encoder.tokens_embedded}")
    print(f"  保守域tokens: {encoder.tokens_in_conserved}")

    # 保存结果
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    output_token_ids = outputs[0].detach().cpu().tolist()
    return final_sequence, output_token_ids, prompt_token_count, encoder, prompt_length, reading_frame, output_path, mapper, conserved_regions, prompt


def load_config(config_path: Optional[str]) -> Dict:
    """加载配置文件"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def create_default_config():
    """创建默认配置文件"""
    config = {
        "model_path": "",
        "reference_parquet": "",
        "reference_protein_id": "",
        "default_length": 1500,
        "default_temperature": 1.0,
        "default_top_k": 10,
        "default_promoter": DEFAULT_PROMOTER,
        "default_reading_frame": 0
    }
    with open("config_adg.json", 'w') as f:
        json.dump(config, f, indent=2)
    print("已创建配置文件: config_adg.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADG DNA隐写术编码器")

    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--create-config", action="store_true")
    parser.add_argument("--model", default=r"C:\huggingface\checkpoint-human", help="模型路径")
    parser.add_argument("--output", default="./outputs", help="输出目录")
    parser.add_argument("--reference-parquet", default=r"C:\huggingface\dataset\cytochrome-p450-cds\atrain.parquet",
                        help="参考序列parquet文件")
    parser.add_argument("--reference-protein-id", default="AAA03751.1", help="参考蛋白质ID")
    parser.add_argument("--secret", help="秘密信息")
    parser.add_argument("--secret-file",
                        default=r".\bit\bit5.txt",
                        help="秘密信息文件")
    parser.add_argument("--length", type=int, default=1500, help="目标长度")
    parser.add_argument("--temperature", type=float, default=1.8, help="温度")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k采样 (默认10)")
    parser.add_argument("--promoter", default="ATGATGACCATCTCTTTGATTTGGGGGATTGCTATGGTAGTG",help="启动子序列")
    parser.add_argument("--reading-frame", type=int, default=0, help="阅读框")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--debug", action="store_true", help="调试模式")

    args = parser.parse_args()

    if args.create_config:
        create_default_config()
        exit(0)

    config = load_config(args.config)

    # 获取secret
    secret = None
    if args.secret:
        secret = args.secret
    elif args.secret_file and os.path.exists(args.secret_file):
        with open(args.secret_file, 'r') as f:
            secret = ''.join(f.read().split())

    if not secret:
        print("错误: 必须指定秘密信息 (--secret 或 --secret-file)")
        exit(1)

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 生成
    final_seq, output_token_ids, prompt_token_count, encoder, prompt_len, rf, output_path, mapper, regions, prompt = generate_with_adg(
        model_path=args.model,
        secret_message=secret,
        output_dir=args.output,
        reference_parquet=args.reference_parquet,
        reference_protein_id=args.reference_protein_id,
        target_length=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        promoter=args.promoter,
        reading_frame=args.reading_frame,
        debug=args.debug
    )

    # 保存
    timestamp = output_path.name

    # FASTA
    fasta_file = output_path / f"sequence_{timestamp}.fasta"
    with open(fasta_file, 'w') as f:
        f.write(f">ADG_Stego|length={len(final_seq)}|bits={encoder.current_bit_index}\n")
        for i in range(0, len(final_seq), 60):
            f.write(final_seq[i:i + 60] + "\n")

    # 元数据 - 包含关键参数供解码器使用
    meta_file = output_path / f"metadata_{timestamp}.txt"
    with open(meta_file, 'w') as f:
        f.write(f"truncated_prompt_length={prompt_len}\n")
        f.write(f"actual_prompt={prompt}\n")
        f.write(f"secret={secret}\n")
        f.write(f"payload_bits={len(secret)}\n")
        f.write(f"embedded_bits={encoder.current_bit_index}\n")
        f.write(f"prompt_token_count={prompt_token_count}\n")
        f.write(f"temperature={args.temperature}\n")
        f.write(f"top_k={args.top_k}\n")  # 重要：保存top_k供解码器使用
        f.write(f"seed={args.seed}\n")
        f.write(f"reading_frame={rf}\n")
        f.write(f"algorithm=HuffmanStego\n")
        if args.reference_parquet:
            f.write(f"reference_parquet={args.reference_parquet}\n")
            f.write(f"reference_protein_id={args.reference_protein_id}\n")

    # 保存精确的输出token序列，供解码器严格对称复现
    token_ids_file = output_path / f"token_ids_{timestamp}.json"
    with open(token_ids_file, 'w', encoding='utf-8') as f:
        json.dump(output_token_ids, f)

    with open(meta_file, 'a', encoding='utf-8') as f:
        f.write(f"token_ids_file={token_ids_file.name}\n")

    print(f"\n结果已保存到: {output_path}")
    print(f"Token IDs已保存到: {token_ids_file}")