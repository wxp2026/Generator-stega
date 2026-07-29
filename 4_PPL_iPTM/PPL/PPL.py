import math

from modeling_progen import ProGenForCausalLM

import torch
from tokenizers import Tokenizer
import torch.nn.functional as F

# 加载模型和tokenizer
model_path = r'./models/progen2-base'
model = ProGenForCausalLM.from_pretrained(model_path)
import os
from tokenizers import Tokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tokenizer = Tokenizer.from_file(os.path.join(BASE_DIR, "tokenizer.json"))


# 移动模型到设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()  # 设置为评估模式


def calculate_perplexity(sequence, model, tokenizer, device):
    """
    计算给定蛋白质序列的困惑度

    Args:
        sequence: 蛋白质序列字符串
        model: ProGen模型
        tokenizer: tokenizer
        device: torch设备

    Returns:
        perplexity: 困惑度值（越低越好）
        avg_loss: 平均损失
    """
    # Tokenize序列
    encoded = tokenizer.encode(sequence)
    input_ids = torch.tensor([encoded.ids]).to(device)

    with torch.no_grad():
        # 获取模型输出
        outputs = model(input_ids, labels=input_ids)

        # 损失已经是平均的负对数似然
        loss = outputs.loss

        # 计算困惑度
        perplexity = torch.pow(2, loss / math.log(2))


    return perplexity.item(), loss.item()


def calculate_perplexity_detailed(sequence, model, tokenizer, device):
    """
    详细计算困惑度，返回更多信息

    Args:
        sequence: 蛋白质序列字符串
        model: ProGen模型
        tokenizer: tokenizer
        device: torch设备

    Returns:
        dict: 包含PPL、损失、每个token的负对数似然等信息
    """
    # Tokenize序列
    encoded = tokenizer.encode(sequence)
    input_ids = torch.tensor([encoded.ids]).to(device)

    with torch.no_grad():
        # 获取模型输出
        outputs = model(input_ids)
        logits = outputs.logits

        # 计算每个位置的损失
        # 移位：预测下一个token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # 计算交叉熵损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # 计算平均损失
        avg_loss = loss_per_token.mean()

        # 计算困惑度
        perplexity = torch.exp(avg_loss)

        # 每个token的困惑度
        token_ppls = torch.exp(loss_per_token)

    return {
        'perplexity': perplexity.item(),
        'avg_loss': avg_loss.item(),
        'token_losses': loss_per_token.cpu().numpy(),
        'token_ppls': token_ppls.cpu().numpy(),
        'sequence_length': len(encoded.ids)
    }


def evaluate_protein_quality(sequence, model, tokenizer, device):
    """
    评估蛋白质序列质量

    Args:
        sequence: 蛋白质序列字符串
        model: ProGen模型
        tokenizer: tokenizer
        device: torch设备

    Returns:
        dict: 包含各种质量指标
    """
    results = calculate_perplexity_detailed(sequence, model, tokenizer, device)

    # 质量评估
    ppl = results['perplexity']

    # 一般来说，PPL越低表示序列质量越好
    # 这里给出一个简单的质量评级（需要根据实际数据调整阈值）
    if ppl < 2.0:
        quality = "Excellent"
    elif ppl < 5.0:
        quality = "Good"
    elif ppl < 10.0:
        quality = "Fair"
    else:
        quality = "Poor"

    print(f"\n=== 蛋白质序列质量评估 ===")
    print(f"序列长度: {results['sequence_length']}")
    print(f"困惑度 (PPL): {ppl:.4f}")
    print(f"平均损失: {results['avg_loss']:.4f}")
    print(f"质量评级: {quality}")
    print(f"最大token PPL: {results['token_ppls'].max():.4f}")
    print(f"最小token PPL: {results['token_ppls'].min():.4f}")

    return results


# 示例使用
if __name__ == "__main__":
    # 测试序列
    target_sequence = ('MMTISLIWGIAMVVMLPIYGFFFLATVRKDVPQDGHEKPPGPFALPFLGNLLQLNFQNPHLSMHQLSKKYGPVFTIHLGPKRMVVLCGYKTVKEALLNHGDEFGDDFKGRPDLYSFNLISNGQIMAFKQDSGSRCLVVSEANVICAMCFGQRYDHDNQELLSIVNLSNESLKAAGSAQHPAVIFNLYPWLGDEMVGKKLIQEFQDLFMQKLIKEHYRTFEKGHIRDLIDSLIKAHKEKKSEEANSEVLKGIVTDLVFADLFGAGFDTVTTAISWSLLLLVNHPEVQRKIQEELDTVIGRNRRPRMSDRAQLPYLEAFILETFRHSSFLPFNIPHSTTRDTSLGGFYIPKGCCVFVNQWQVNHDPELWVDPNNFRPERFLTPSGTVDKVLSEKVILFGMGKRKCVGETIGRWEVFLFLAILLQQVEFSVSPGEKVDITPIYGLTLKYARCEHFQVQTRSF-RAVCGAQEPGYPQ-KLLTEEN-NVY-FYYYYYMIAIISPL')
# 计算PPL

    ppl, loss = calculate_perplexity(target_sequence, model, tokenizer, device)
    print(f"困惑度: {ppl:.4f}")
    print(f"损失: {loss:.4f}")

