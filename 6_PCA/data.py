import re

def split_each_line_to_198bp(input_path=r".\5_PCA _ SWD\CCRS.txt", output_path=r".\5_PCA _ SWD\CCRS_clean.txt", line_len=198):
    out_lines = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            # 只保留 A/C/G/T（如果你想保留N，把ACGT改成ACGTN）
            seq = re.sub(r"[^ACGT]", "", line.upper())

            if len(seq) < line_len:
                continue

            full_chunks = len(seq) // line_len
            for i in range(full_chunks):
                start = i * line_len
                fout.write(seq[start:start + line_len] + "\n")
                out_lines += 1

    print(f"Done. 输出行数: {out_lines}, 输出文件: {output_path}")

if __name__ == "__main__":
    split_each_line_to_198bp(r".\5_PCA _ SWD\CCRS.txt",
                             r"/DNA-Synthetic-Steganography-Based-on-Conditional-Probability-Adaptive-Coding-main/5_PCA _ SWD/output/CCRS_clean.txt", 198)
