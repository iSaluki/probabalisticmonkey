"""Benchmark for Probabilistic Monkey, run against the shipped model in model/.

    python eval/benchmark.py [--out eval/results.md]

1. Multiple choice (computers & networks): each option is scored by the
   average log-probability the model gives it as an answer; the best wins.
2. Open-ended (computers & networks): the model answers in its own words
   (greedy decoding); an answer passes if it mentions a key concept.
3. English: free-form prompts whose replies are written to the report for
   reading. These are judged by a reviewer, not by a script.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "app"))
from engine import Monkey  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results.md"))
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    spec = json.load(open(os.path.join(HERE, "questions.json")))
    m = Monkey(a.model) if a.model else Monkey()
    t0 = time.time()
    lines = ["# Probabilistic Monkey benchmark results", ""]

    mc_ok = 0
    mc_lines = []
    for item in spec["multiple_choice"]:
        msgs = [{"role": "user", "content": item["q"]}]
        scores = [m.score(msgs, o) for o in item["options"]]
        pick = max(range(len(scores)), key=lambda i: scores[i])
        ok = pick == item["answer"]
        mc_ok += ok
        mc_lines.append(f"| {'PASS' if ok else 'FAIL'} | {item['q']} | {item['options'][pick]} | {item['options'][item['answer']]} |")
    mc = mc_ok / len(spec["multiple_choice"])
    print(f"multiple choice: {mc_ok}/{len(spec['multiple_choice'])}", flush=True)

    oe_ok = 0
    oe_lines = []
    for item in spec["open_ended"]:
        ans = m.reply([{"role": "user", "content": item["q"]}], temperature=0, max_new_tokens=120)
        ok = any(k in ans.lower() for k in item["any"])
        oe_ok += ok
        oe_lines += [f"**{'PASS' if ok else 'FAIL'} - {item['q']}**", "", "> " + ans.replace("\n", "\n> "), ""]
        print(("PASS" if ok else "FAIL"), item["q"], "->", ans[:200].replace("\n", " "), flush=True)
    oe = oe_ok / len(spec["open_ended"])
    print(f"open ended: {oe_ok}/{len(spec['open_ended'])}", flush=True)

    en_lines = []
    for q in spec["english"]:
        ans = m.reply([{"role": "user", "content": q}], temperature=0.7, max_new_tokens=160, seed=1)
        en_lines += [f"**{q}**", "", "> " + ans.replace("\n", "\n> "), ""]
        print("EN", q, "->", ans[:300].replace("\n", " "), flush=True)

    crit = spec["pass_criteria"]
    passed = mc >= crit["multiple_choice_min"] and oe >= crit["open_ended_min"]
    lines += [
        f"- Multiple choice: **{mc_ok}/{len(spec['multiple_choice'])} ({mc:.0%})**, required {crit['multiple_choice_min']:.0%}",
        f"- Open-ended: **{oe_ok}/{len(spec['open_ended'])} ({oe:.0%})**, required {crit['open_ended_min']:.0%}",
        f"- Automatic verdict: **{'PASS' if passed else 'FAIL'}**",
        f"- Runtime: {time.time() - t0:.0f}s on CPU with the NumPy engine",
        "", "## Multiple choice", "", "| result | question | model's pick | correct |", "|---|---|---|---|",
        *mc_lines, "", "## Open-ended (greedy decoding)", "", *oe_lines,
        "## English generation (sampled, temperature 0.7)", "", *en_lines,
    ]
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("verdict:", "PASS" if passed else "FAIL", "- report written to", a.out)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
