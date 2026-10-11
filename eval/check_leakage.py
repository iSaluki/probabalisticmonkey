"""Checks that no benchmark question appears in the hand-written training data.

A question that appears verbatim fails the check. A question that shares a run
of six consecutive words with the training text is reported as a warning so it
can be inspected and disclosed (it may just be a common phrase).

The large public corpora are not checked line by line, but they were never
shown the benchmark either (the benchmark was written for this project).
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "training", "data")


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


spec = json.load(open(os.path.join(HERE, "questions.json")))
questions = [i["q"] for i in spec["multiple_choice"] + spec["open_ended"]] + spec["english"]
train = norm(" ".join(open(os.path.join(DATA, f)).read() for f in os.listdir(DATA)).replace("\n", " "))
train = re.sub(r" +", " ", train)
N = 6
train_words = train.split()
train_ngrams = {" ".join(train_words[i:i + N]) for i in range(len(train_words) - N + 1)}


def shared_phrases(q):
    w = norm(q).split()
    return sorted({" ".join(w[i:i + N]) for i in range(len(w) - N + 1)} & train_ngrams)


leaks = [q for q in questions if norm(q) in train]
for q in leaks:
    print("LEAK (verbatim):", q)
warnings = 0
for q in questions:
    shared = shared_phrases(q)
    if shared and q not in leaks:
        warnings += 1
        print("WARNING (shared phrase):", q, "->", "; ".join(f'"{p}"' for p in shared))
print(f"{len(questions)} benchmark questions checked: {len(leaks)} verbatim leaks, {warnings} shared-phrase warnings")
sys.exit(1 if leaks else 0)
