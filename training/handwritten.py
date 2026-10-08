"""Loads the hand-written teaching material in training/data/."""
import os
import random
import re

from common import HANDWRITTEN


def load_notes():
    """Return the technical notes split into passages (one per section)."""
    text = open(os.path.join(HANDWRITTEN, "tech_notes.md")).read()
    parts = re.split(r"\n(?=#{1,2} )", text)
    return [p.strip() for p in parts if p.strip()]


def load_qa():
    """Return a list of (questions, answer) blocks."""
    blocks = []
    qs, ans = [], None
    for line in open(os.path.join(HANDWRITTEN, "tech_qa.txt")):
        line = line.rstrip("\n")
        if line.startswith("#"):
            continue
        if line.startswith("Q:"):
            qs.append(line[2:].strip())
        elif line.startswith("A:"):
            ans = line[2:].strip()
        elif not line.strip() and qs and ans:
            blocks.append((qs, ans))
            qs, ans = [], None
    if qs and ans:
        blocks.append((qs, ans))
    return blocks


def vary(q, rng):
    """Light augmentation so the model copes with casual typing."""
    r = rng.random()
    if r < 0.2:
        q = q.lower()
    elif r < 0.3:
        q = q.rstrip("?!.")
    elif r < 0.4:
        q = q.lower().rstrip("?!.")
    return q


def qa_conversations(copies, seed=0):
    """Single and multi-turn chat conversations built from the Q/A blocks."""
    rng = random.Random(seed)
    blocks = load_qa()
    convs = []
    for _ in range(copies):
        for qs, a in blocks:
            for q in qs:
                convs.append([{"role": "user", "content": vary(q, rng)}, {"role": "assistant", "content": a}])
        # multi-turn: chain a few random exchanges so the model learns to use history
        for _ in range(len(blocks) // 2):
            conv = []
            for qs, a in rng.sample(blocks, rng.randint(2, 4)):
                conv += [{"role": "user", "content": vary(rng.choice(qs), rng)}, {"role": "assistant", "content": a}]
            convs.append(conv)
    rng.shuffle(convs)
    return convs
