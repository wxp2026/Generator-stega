import random

def random_dna(length=1500, seed=None):
    if seed is not None:
        random.seed(seed)
    bases = "ATGC"
    return "".join(random.choice(bases) for _ in range(length))

seq = random_dna(1500, seed=42)  # seed 可删掉
print(seq)
print("Length:", len(seq))
