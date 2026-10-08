"""Checks that no benchmark question appears in the hand-written training data.

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
leaks = [q for q in questions if norm(q) in train]
for q in leaks:
    print("LEAK:", q)
print(f"{len(questions)} benchmark questions checked, {len(leaks)} found verbatim in training data")
sys.exit(1 if leaks else 0)
