import torch
import torch.nn as nn


class LM(nn.Module):
    def __init__(self, vocab_size, embed_size=350, hidden_dim=512, num_layers=2, dropout_rate=0.2):
        super(LM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # === 修复点 ===
        # 将 dim=2 改为 dim=-1。
        # dim=-1 表示在张量的“最后一个维度”上进行计算，这样兼容 2D 和 3D 输入。
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        # =============

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embed = self.embedding(x)  # [batch, seq, embed_dim]
        output, (hn, cn) = self.lstm(embed)  # [batch, seq, hidden_dim]
        logits = self.fc(output)  # [batch, seq, vocab_size]
        return self.log_softmax(logits)

    def get_prob(self, x):
        """获取用于隐写的概率分布 (非 log)"""
        embed = self.embedding(x)
        output, _ = self.lstm(embed)
        logits = self.fc(output)

        # 取最后一个时间步
        last_step_logits = logits[:, -1, :]  # Shape: [batch, vocab_size] (2D Tensor)

        # 这里之前是 dim=2 导致报错，现在 dim=-1 会自动识别为 dim=1
        return self.softmax(last_step_logits)