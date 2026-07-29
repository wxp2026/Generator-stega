import torch
from stega import embed_secret
from utils import Vocabulary
from lm import LM

import torch
from utils import Vocabulary   # 仅导入Vocabulary
from lm import LM             # LM在lm.py中
from stega import embed_secret

import torch
from utils import Vocabulary
from lm import LM
from stega import embed_secret

# -----------------------------
# 配置
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seq_len = 198   # pseudo-sequence长度
model_path = r'.\3_Model_training & Stega_generating\adg_train\model.pth'
reference_files = [r'.\4_Baselines\AAA03751.1.txt', r'.\4_Baselines\AAA36405.1.txt']
output_file = r'.\4_Baselines\output\LSTM-ADG\DNA.txt'

# -----------------------------
# 加载词表和模型
# -----------------------------
vocab = Vocabulary(reference_files)
# 注意这里参数名改成 embed_size
model = LM(vocab.vocab_size, embed_size=128, hidden_dim=256).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -----------------------------
# 生成伪序列
# -----------------------------
all_sequences = []

for ref_file in reference_files:
    with open(ref_file, 'r') as f:
        ref_seq = ''.join([line.strip() for line in f.readlines()])

    context = [vocab.w2i['_BOS']]
    secret_bits = [0,1]*100  # 示例比特流
    generated_ids, _ = embed_secret(model, vocab, context, secret_bits, device, target_len=seq_len)
    generated_seq = ''.join([vocab.i2w[i] for i in generated_ids if i in vocab.i2w])
    all_sequences.append(generated_seq)

# -----------------------------
# 保存到DNA.txt
# -----------------------------
with open(output_file, 'w') as f:
    for seq in all_sequences:
        f.write(seq + '\n')

print(f"生成完成，共 {len(all_sequences)} 条伪 DNA 序列，已保存到 {output_file}")
