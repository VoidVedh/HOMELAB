import json
import os
from typing import Dict, List, Tuple

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<sys>": 4,
    "<usr>": 5,
    "<bot>": 6,
}

class BPETokenizer:
    def __init__(self, vocab: Dict[str, int] = None, merges: List[Tuple[str, str]] = None):
        self.special_tokens = dict(SPECIAL_TOKENS)
        self.vocab = vocab or dict(self.special_tokens)
        self.merges = merges or []
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _get_stats(self, words_counts: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        stats = {}
        for word, count in words_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                stats[pair] = stats.get(pair, 0) + count
        return stats

    def _merge_word(self, word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def train(self, text: str, vocab_size: int = 4096):
        if vocab_size < len(self.special_tokens) + 256:
            vocab_size = len(self.special_tokens) + 256

        # Initialize base byte characters
        base_bytes = [chr(i) for i in range(256)]
        for b in base_bytes:
            if b not in self.vocab:
                idx = len(self.vocab)
                self.vocab[b] = idx

        # Word frequency counting
        raw_words = text.split()
        words_counts = {}
        for w in raw_words:
            tup = tuple(list(w))
            words_counts[tup] = words_counts.get(tup, 0) + 1

        num_merges = vocab_size - len(self.vocab)
        for _ in range(num_merges):
            stats = self._get_stats(words_counts)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 2:
                break
            
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
            self.merges.append(best_pair)

            new_words_counts = {}
            for word, count in words_counts.items():
                new_word = self._merge_word(word, best_pair)
                new_words_counts[new_word] = count
            words_counts = new_words_counts

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if not text:
            return []
        
        tokens = []
        if add_special_tokens:
            tokens.append(self.special_tokens["<bos>"])

        raw_words = text.split(" ")
        for i, word_str in enumerate(raw_words):
            if i > 0:
                word_str = " " + word_str
            word = tuple(list(word_str))
            for pair in self.merges:
                word = self._merge_word(word, pair)
            for char_tok in word:
                tokens.append(self.vocab.get(char_tok, self.special_tokens["<unk>"]))

        if add_special_tokens:
            tokens.append(self.special_tokens["<eos>"])

        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        res = []
        special_ids = set(self.special_tokens.values()) if skip_special_tokens else set()
        for tok in tokens:
            if tok in special_ids:
                continue
            res.append(self.inverse_vocab.get(tok, "<unk>"))
        return "".join(res)

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab": self.vocab,
                "merges": self.merges,
                "special_tokens": self.special_tokens
            }, f, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "BPETokenizer":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merges = [tuple(pair) for pair in data["merges"]]
        tokenizer = cls(vocab=data["vocab"], merges=merges)
        return tokenizer
