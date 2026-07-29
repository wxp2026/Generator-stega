# 打开原始文件和输出文件
with open(r".\5_PCA _ SWD\output\raw_clean.txt", "r") as infile, open(r".\3_Model_training & Stega_generating\adg_train\data\raw.txt", "w") as outfile:
    for line in infile:
        seq = line.strip()  # 去掉空白字符和换行
        # 计算能分成多少完整的198bp段
        num_segments = len(seq) // 198
        for s in range(num_segments):
            segment = seq[s*198:(s+1)*198]  # 取198bp
            # 每6bp分组
            chunks = [segment[i:i+6] for i in range(0, 198, 6)]
            formatted_line = " ".join(chunks)
            outfile.write(formatted_line + "\n")
