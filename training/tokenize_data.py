"""Step 4: turn the corpus into flat uint16 token files for fast training.

  pretrain.bin   shuffled mix of all documents, each followed by <|endoftext|>
  val.bin        held-out documents for measuring validation loss
  sft.bin / sft_mask.bin
                 chat conversations packed into rows of SEQ+1 tokens; the mask
                 is 1 on the assistant's words (the only tokens trained on)
"""
import json
import os
import random

import numpy as np
from tokenizers import Tokenizer

from common import ASSISTANT, CORPUS, END, EOT, MODEL_DIR, TOKENS, USER, VARIANTS, format_chat
from handwritten import BIG_QA_FILES, load_notes, qa_conversations

SEQ = 512
rng = random.Random(42)
tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
EOT_ID = tok.token_to_id(EOT)


def encode_many(texts):
    out = []
    for i in range(0, len(texts), 10000):
        out += [np.array(e.ids + [EOT_ID], dtype=np.uint16) for e in tok.encode_batch(texts[i:i + 10000])]
    return out


def read_jsonl(name):
    with open(os.path.join(CORPUS, name)) as f:
        for line in f:
            yield json.loads(line)


def build_pretrain():
    docs = []
    val = []
    for name in ["cosmopedia.jsonl", "fineweb.jsonl"]:
        rows = list(read_jsonl(name))
        rng.shuffle(rows)
        val += [d["text"] for d in rows[:1000]]
        # technical documents are seen twice: this skews the model towards technical topics
        texts = [d["text"] for d in rows[1000:] for _ in range(2 if d["tech"] else 1)]
        enc = encode_many(texts)
        print(name, len(enc), "docs", sum(map(len, enc)) / 1e6, "M tokens", flush=True)
        docs += enc
    chats = [format_chat(d["messages"]) for d in read_jsonl("chat.jsonl")]
    enc = encode_many(chats)
    print("chat", len(enc), "docs", sum(map(len, enc)) / 1e6, "M tokens", flush=True)
    docs += enc
    hand = load_notes() * 40 + [format_chat(c) for c in qa_conversations(25, seed=1)]
    enc = encode_many(hand)
    print("handwritten", len(enc), "docs", sum(map(len, enc)) / 1e6, "M tokens", flush=True)
    docs += enc
    rng.shuffle(docs)
    arr = np.concatenate(docs)
    arr.tofile(os.path.join(TOKENS, "pretrain.bin"))
    np.concatenate(encode_many(val)).tofile(os.path.join(TOKENS, "val.bin"))
    print("pretrain tokens %.1fM" % (len(arr) / 1e6))


def encode_chat_masked(messages):
    ids, mask = [], []
    for m in messages:
        if m["role"] == "user":
            t = tok.encode(f"{USER}{m['content'].strip()}{END}").ids
            ids += t
            mask += [0] * len(t)
        else:
            h = tok.encode(ASSISTANT).ids
            t = tok.encode(f"{m['content'].strip()}{END}").ids
            ids += h + t
            mask += [0] * len(h) + [1] * len(t)
    return ids + [EOT_ID], mask + [0]


def build_sft(max_chat=90000, seq=SEQ, out_dir=TOKENS, extra=None, chat_files=("chat.jsonl",)):
    chats = [d["messages"] for f in chat_files for d in read_jsonl(f)]
    rng.shuffle(chats)
    convs = chats[:max_chat] + (extra if extra is not None else qa_conversations(12, seed=2))
    rng.shuffle(convs)
    rows, masks = [], []
    cur, curm = [], []
    for c in convs:
        ids, m = encode_chat_masked(c)
        if len(ids) > seq + 1:
            ids, m = ids[:seq + 1], m[:seq + 1]
        if len(cur) + len(ids) > seq + 1:
            pad = seq + 1 - len(cur)
            rows.append(cur + [EOT_ID] * pad)
            masks.append(curm + [0] * pad)
            cur, curm = [], []
        cur += ids
        curm += m
    rows = np.array(rows, dtype=np.uint16)
    masks = np.array(masks, dtype=np.uint8)
    rows.tofile(os.path.join(out_dir, "sft.bin"))
    masks.tofile(os.path.join(out_dir, "sft_mask.bin"))
    print("sft rows", len(rows), "trained tokens %.1fM" % (masks.sum() / 1e6))


def build_big():
    """Probabilistic Monkey Big: all corpora, longer chats, processes and generated skills."""
    import synthetic
    out_dir = VARIANTS["big"]["tokens"]
    seq = VARIANTS["big"]["model"]["max_seq_len"]
    docs, val = [], []
    for name in ["cosmopedia.jsonl", "fineweb.jsonl", "cosmopedia_big.jsonl", "fineweb_big.jsonl"]:
        rows = list(read_jsonl(name))
        rng.shuffle(rows)
        val += [d["text"] for d in rows[:500]]
        texts = [d["text"] for d in rows[500:] for _ in range(2 if d["tech"] else 1)]
        enc = encode_many(texts)
        print(name, len(enc), "docs", sum(map(len, enc)) / 1e6, "M tokens", flush=True)
        docs += enc
        del rows, texts
    chats = [format_chat(d["messages"]) for f in ("chat.jsonl", "chat_big.jsonl") for d in read_jsonl(f)]
    enc = encode_many(chats)
    print("chat", len(enc), "docs", sum(map(len, enc)) / 1e6, "M tokens", flush=True)
    docs += enc
    hand = load_notes() * 40 + [format_chat(c) for c in qa_conversations(25, seed=1, files=BIG_QA_FILES)]
    hand += [format_chat(c) for c in synthetic.conversations(150_000, seed=11)]
    enc = encode_many(hand)
    print("handwritten + generated", len(enc), "docs", sum(map(len, enc)) / 1e6, "M tokens", flush=True)
    docs += enc
    rng.shuffle(docs)
    arr = np.concatenate(docs)
    arr.tofile(os.path.join(out_dir, "pretrain.bin"))
    np.concatenate(encode_many(val)).tofile(os.path.join(out_dir, "val.bin"))
    print("big pretrain tokens %.1fM" % (len(arr) / 1e6), flush=True)
    del docs, arr
    extra = qa_conversations(12, seed=2, files=BIG_QA_FILES) + synthetic.conversations(40_000, seed=12)
    build_sft(max_chat=150_000, seq=seq, out_dir=out_dir, extra=extra, chat_files=("chat.jsonl", "chat_big.jsonl"))


if __name__ == "__main__":
    import sys
    parts = sys.argv[1:] or ["sft", "pretrain"]
    if "big" in parts:
        build_big()
    if "sft" in parts:
        build_sft()
    if "pretrain" in parts:
        build_pretrain()
