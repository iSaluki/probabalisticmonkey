"""Checks that no benchmark question appears in the teaching data written for this project.

Checked: every hand-written file in training/data/ and, for Probabilistic Gorilla,
the exact generated conversations from training/synthetic.py.

A question that appears verbatim fails the check. A question that shares a run
of six consecutive words with the teaching data is reported as a warning so it
can be inspected and disclosed (it may just be a common phrase).

The large public corpora are not checked line by line, but they were never
shown the benchmarks either (the benchmarks were written for this project).
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "training", "data")
N = 6


def norm(s):
    return re.sub(r" +", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def ngrams(text):
    w = text.split()
    return {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)}


def monkey_questions():
    spec = json.load(open(os.path.join(HERE, "questions.json")))
    return [i["q"] for i in spec["multiple_choice"] + spec["open_ended"]] + spec["english"]


def gorilla_questions():
    spec = json.load(open(os.path.join(HERE, "gorilla_questions.json")))
    qs = [i["q"] for k in ("multiple_choice", "troubleshooting", "processes", "calculations") for i in spec[k]]
    qs += [t[0] for m in spec["memory"] for t in m["turns"]]
    return qs + spec["english"]


def check(name, questions, texts):
    train = norm(" ".join(texts))
    grams = ngrams(train)
    leaks = [q for q in questions if f" {norm(q)} " in f" {train} "]
    warnings = 0
    for q in leaks:
        print(f"  LEAK (verbatim): {q}")
    for q in questions:
        shared = sorted(ngrams(norm(q)) & grams)
        if shared and q not in leaks:
            warnings += 1
            print(f'  WARNING (shared phrase): {q} -> {"; ".join(shared[:3])}')
    print(f"{name}: {len(questions)} questions checked, {len(leaks)} verbatim leaks, {warnings} shared-phrase warnings")
    return not leaks


if __name__ == "__main__":
    hand = [open(os.path.join(DATA, f)).read() for f in sorted(os.listdir(DATA))]
    ok = check("Monkey benchmark vs hand-written data", monkey_questions(), hand)
    sys.path.insert(0, os.path.join(REPO, "training"))
    import synthetic  # noqa: E402
    generated = [m["content"] for seed, n in ((11, 150_000), (12, 40_000))
                 for c in synthetic.conversations(n, seed=seed) for m in c]
    ok &= check("Gorilla benchmark vs hand-written + generated data", gorilla_questions(), hand + generated)
    sys.exit(0 if ok else 1)
