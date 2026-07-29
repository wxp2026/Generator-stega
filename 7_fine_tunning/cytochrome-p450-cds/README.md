---
task_categories:
- text-generation
tags:
- biology
- genomics
---

## About

The protein sequences and protein-coding DNA sequences for the **Cytochrome P450 family** were sourced from [UniProt](https://www.uniprot.org/) and [NCBI](https://www.ncbi.nlm.nih.gov/).  

Protein-to-DNA pairing was achieved by cross-referencing `protein_ids` and `gene_ids`.

## How to use
```python
from datasets import load_dataset

datasets = load_dataset("GenerTeam/cytochrome-p450-cds")
```