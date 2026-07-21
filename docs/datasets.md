# Datasets & DataLoader Specification

## Datasets Strategy
`ai_core.datasets` handles dataset acquisition, text preprocessing, tokenization caching, and streaming window batching.

### Sequence Window Batching
For input sequence length $L$, the DataLoader constructs input tensor $X \in \mathbb{Z}^{B \times L}$ and target tensor $Y \in \mathbb{Z}^{B \times L}$ where $Y_{i, t} = X_{i, t+1}$.

```
Input (X):  [ <bos>, The, quick, brown, fox ]
Target (Y): [ The, quick, brown, fox,  <eos> ]
```

## Corpus Formatting
Text corpora are structured as clean UTF-8 text files stored in `data/` or `ai_core/datasets/samples/`. Tokenized datasets are cached as binary `.npy` / `.bin` arrays for rapid memory-mapped reading.
