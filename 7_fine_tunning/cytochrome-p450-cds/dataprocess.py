import pandas as pd
import numpy as np

# 读取数据
df = pd.read_parquet(r"C:\huggingface\dataset\cytochrome-p450-cds\train.parquet")

# 设置随机种子
np.random.seed(42)

# 生成随机数
random_col = np.random.random(len(df))

# 按比例分割
train_mask = random_col < 0.8
val_mask = (random_col >= 0.8) & (random_col < 0.9)
test_mask = random_col >= 0.9

train_df = df[train_mask]
val_df = df[val_mask]
test_df = df[test_mask]

print(f"训练集数量: {len(train_df)}")
print(f"验证集数量: {len(val_df)}")
print(f"测试集数量: {len(test_df)}")

# 保存结果
train_df.to_parquet(r"C:\huggingface\dataset\cytochrome-p450-cds\train_split.parquet")
val_df.to_parquet(r"C:\huggingface\dataset\cytochrome-p450-cds\val_split.parquet")
test_df.to_parquet(r"C:\huggingface\dataset\cytochrome-p450-cds\test_split.parquet")

