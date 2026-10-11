"""Benchmark for Probabilistic Monkey Big (the harder task), run against model_big/.

    python eval/big_benchmark.py [--model model_big] [--out eval/big_results.md]

Sections (all questions in eval/big_questions.json, written before training):
  * multiple choice: scored like the small benchmark, by the average
    log-probability of each bare option as the reply (options not shown)
  * troubleshooting: greedy answer must name the likely cause (keywords)
  * processes: a step-by-step explanation must cover enough key steps
  * calculations: the last number in the answer must be correct
    (yes/no questions: the answer must start with the right word)
  * memory: a fact is given early in a conversation; the final reply must
    repeat it (the model's own replies are used for the earlier turns)
  * English: longer free-form writing, saved to the report for reading
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "app"))
from engine import Monkey  # noqa: E402


def last_number(text):
    nums = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def quote(text):
    return "> " + text.replace("\n", "\n> ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(REPO, "model_big"))
    ap.add_argument("--out", default=os.path.join(HERE, "big_results.md"))
    a = ap.parse_args()
    spec = json.load(open(os.path.join(HERE, "big_questions.json")))
    m = Monkey(a.model)
    t0 = time.time()
    scores, report = {}, []

    def greedy(messages, n=200):
        return m.reply(messages, temperature=0, max_new_tokens=n)

    # multiple choice
    ok, rows = 0, []
    for item in spec["multiple_choice"]:
        msgs = [{"role": "user", "content": item["q"]}]
        s = [m.score(msgs, o) for o in item["options"]]
        pick = max(range(len(s)), key=lambda i: s[i])
        ok += pick == item["answer"]
        rows.append(f"| {'PASS' if pick == item['answer'] else 'FAIL'} | {item['q']} | {item['options'][pick]} | "
                    f"{item['options'][item['answer']]} |")
    scores["multiple_choice"] = (ok, len(spec["multiple_choice"]))
    report += ["## Multiple choice", "", "| result | question | model's pick | correct |", "|---|---|---|---|", *rows, ""]

    # troubleshooting
    ok, lines = 0, []
    for item in spec["troubleshooting"]:
        ans = greedy([{"role": "user", "content": item["q"]}])
        hit = any(k in ans.lower() for k in item["any"])
        ok += hit
        lines += [f"**{'PASS' if hit else 'FAIL'} - {item['q']}**", "", quote(ans), ""]
    scores["troubleshooting"] = (ok, len(spec["troubleshooting"]))
    report += ["## Troubleshooting", "", *lines]

    # processes
    ok, lines = 0, []
    for item in spec["processes"]:
        ans = greedy([{"role": "user", "content": item["q"]}], 300)
        low = ans.lower()
        covered = sum(any(k in low for k in group) for group in item["concepts"])
        hit = covered >= item["min_concepts"]
        ok += hit
        lines += [f"**{'PASS' if hit else 'FAIL'} ({covered} key steps) - {item['q']}**", "", quote(ans), ""]
    scores["processes"] = (ok, len(spec["processes"]))
    report += ["## Step-by-step explanations", "", *lines]

    # calculations
    ok, rows = 0, []
    for item in spec["calculations"]:
        ans = greedy([{"role": "user", "content": item["q"]}], 120)
        if item["answer"] in ("yes", "no"):
            got = (re.findall(r"[a-z]+", ans.lower()) or [""])[0]
        else:
            got = last_number(ans)
        hit = got == item["answer"]
        ok += hit
        rows.append(f"| {'PASS' if hit else 'FAIL'} | {item['q']} | {item['answer']} | {ans.replace('|', '/')} |")
    scores["calculations"] = (ok, len(spec["calculations"]))
    report += ["## Calculations", "", "| result | question | expected | model's answer |", "|---|---|---|---|", *rows, ""]

    # memory
    ok, lines = 0, []
    for item in spec["memory"]:
        conv = []
        for user, _ in item["turns"]:
            conv.append({"role": "user", "content": user})
            conv.append({"role": "assistant", "content": greedy(conv, 120)})
        hit = item["answer"] in conv[-1]["content"].lower()
        ok += hit
        lines += [f"**{'PASS' if hit else 'FAIL'} - expected '{item['answer']}'**", ""]
        lines += [f"- *user:* {c['content']}" if c["role"] == "user" else f"- *monkey:* {c['content']}" for c in conv]
        lines.append("")
    scores["memory"] = (ok, len(spec["memory"]))
    report += ["## Conversation memory", "", *lines]

    # English
    lines = []
    for q in spec["english"]:
        ans = m.reply([{"role": "user", "content": q}], temperature=0.7, max_new_tokens=300, seed=1)
        lines += [f"**{q}**", "", quote(ans), ""]
    report += ["## English writing (sampled, temperature 0.7)", "", *lines]

    crit = spec["pass_criteria"]
    summary, passed = [], True
    for k, (ok, n) in scores.items():
        need = crit[f"{k}_min"]
        good = ok / n >= need
        passed &= good
        summary.append(f"| {k.replace('_', ' ')} | {ok}/{n} ({ok / n:.0%}) | {need:.0%} | {'pass' if good else 'FAIL'} |")
        print(f"{k}: {ok}/{n}", flush=True)
    head = ["# Probabilistic Monkey Big benchmark results", "",
            "| section | score | required | |", "|---|---|---|---|", *summary, "",
            f"Automatic verdict: **{'PASS' if passed else 'FAIL'}** (English is judged by reading the samples below). "
            f"Runtime {time.time() - t0:.0f}s with the NumPy engine.", ""]
    with open(a.out, "w") as f:
        f.write("\n".join(head + report) + "\n")
    print("verdict:", "PASS" if passed else "FAIL", "- report written to", a.out)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
