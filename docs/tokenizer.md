# Tokenizer Architecture & Design

## Overview
`ai_core.tokenizer.BPETokenizer` implements a Byte-Pair Encoding subword tokenization algorithm built natively without third-party black-box wrappers.

## Vocabulary & Special Tokens
The tokenizer reserves special control tokens for sequence formatting and instruction tuning:

| Token | ID | Usage |
|---|---|---|
| `<pad>` | 0 | Padding sequence alignment |
| `<unk>` | 1 | Unknown / out-of-vocab characters |
| `<bos>` | 2 | Beginning of sequence |
| `<eos>` | 3 | End of sequence |
| `<sys>` | 4 | System prompt marker |
| `<usr>` | 5 | User turn marker |
| `<bot>` | 6 | Assistant response marker |

## Training Algorithm
1. Initialize vocabulary with individual characters + special tokens.
2. Iteratively count adjacent pair frequencies in corpus.
3. Merge the most frequent character pair into a new subword token.
4. Repeat until desired vocabulary size (e.g. 4096 for Jarvis-0) is reached.
5. Save learned merge rules and vocabulary mapping to JSON.
