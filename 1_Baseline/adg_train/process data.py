import random

# 文件路径
raw_file = r".\3_Model_training & Stega_generating\adg_train\data\raw.txt"
train_file = r".\3_Model_training & Stega_generating\adg_train\data\train.txt"
test_file = r".\3_Model_training & Stega_generating\adg_train\data\test.txt"

# 读取数据
with open(raw_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 打乱数据
random.shuffle(lines)

# 计算划分位置 (9:1)
split_index = int(len(lines) * 0.9)

train_data = lines[:split_index]
test_data = lines[split_index:]

# 写入 train.txt
with open(train_file, "w", encoding="utf-8") as f:
    f.writelines(train_data)

# 写入 test.txt
with open(test_file, "w", encoding="utf-8") as f:
    f.writelines(test_data)

print("总数据量:", len(lines))
print("train.txt:", len(train_data))
print("test.txt:", len(test_data))
