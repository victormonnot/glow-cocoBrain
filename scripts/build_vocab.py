"""
build_vocab.py — Construit le vocabulaire à partir des captions COCO
=====================================================================
Usage: python scripts/build_vocab.py [--data_root data/coco] [--vocab_size 10000]

Produit: data/coco/vocab.json
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tokenizer import SimpleTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/coco")
    parser.add_argument("--vocab_size", type=int, default=10000)
    args = parser.parse_args()

    ann_path = os.path.join(args.data_root, "annotations", "captions_train2017.json")
    print(f"Loading annotations from {ann_path}...")
    with open(ann_path) as f:
        annotations = json.load(f)

    captions = [ann["caption"] for ann in annotations["annotations"]]
    print(f"Found {len(captions)} captions")

    print(f"Building vocab (max {args.vocab_size} words)...")
    tokenizer = SimpleTokenizer.build_from_captions(captions, args.vocab_size)

    out_path = os.path.join(args.data_root, "vocab.json")
    tokenizer.save(out_path)
    print(f"Vocab saved to {out_path} ({tokenizer.vocab_size} tokens)")

    # Quick sanity check
    sample = captions[0]
    encoded = tokenizer.encode(sample, max_len=32)
    decoded = tokenizer.decode(encoded)
    print(f"\nSanity check:")
    print(f"  Original: {sample}")
    print(f"  Encoded:  {encoded[:15].tolist()}...")
    print(f"  Decoded:  {decoded}")


if __name__ == "__main__":
    main()
