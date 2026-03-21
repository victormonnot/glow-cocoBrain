"""
tokenizer.py — Tokenizer word-level pour COCO Captions
========================================================
Vocabulaire construit à partir des captions d'entraînement.
Tokens spéciaux : <pad>=0, <unk>=1, <bos>=2, <eos>=3.
"""

import json
import re
from collections import Counter
from typing import List

import torch


PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class SimpleTokenizer:
    """Tokenizer word-level simple pour COCO Captions."""

    def __init__(self, word_to_idx: dict, idx_to_word: list):
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.vocab_size = len(idx_to_word)

    @classmethod
    def build_from_captions(
        cls, captions: List[str], max_vocab_size: int = 10000
    ) -> "SimpleTokenizer":
        counter = Counter()
        for caption in captions:
            tokens = _tokenize(caption)
            counter.update(tokens)

        # Top words (reserve 4 slots for special tokens)
        most_common = counter.most_common(max_vocab_size - len(SPECIAL_TOKENS))

        idx_to_word = list(SPECIAL_TOKENS) + [word for word, _ in most_common]
        word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}

        return cls(word_to_idx, idx_to_word)

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        """Texte → tensor de token IDs, padé/tronqué à max_len."""
        words = _tokenize(text)
        ids = [BOS_IDX] + [self.word_to_idx.get(w, UNK_IDX) for w in words] + [EOS_IDX]

        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [PAD_IDX] * (max_len - len(ids))

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Tensor de token IDs → texte lisible."""
        words = []
        for idx in token_ids.tolist():
            if idx == EOS_IDX or idx == PAD_IDX:
                break
            if idx in (BOS_IDX,):
                continue
            if idx < len(self.idx_to_word):
                words.append(self.idx_to_word[idx])
            else:
                words.append("<unk>")
        return " ".join(words)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"idx_to_word": self.idx_to_word}, f)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        with open(path, "r") as f:
            data = json.load(f)
        idx_to_word = data["idx_to_word"]
        word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}
        return cls(word_to_idx, idx_to_word)


def _tokenize(text: str) -> List[str]:
    """Lowercase + split sur whitespace/ponctuation."""
    text = text.lower().strip()
    # Sépare la ponctuation des mots
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()
