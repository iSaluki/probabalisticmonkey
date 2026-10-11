"""Step 2: select and clean documents, skewing the mix towards technical text.

Writes JSONL files into $PM_WORK/corpus:
  cosmopedia.jsonl   synthetic textbooks (all technical ones + a sample of the rest)
  fineweb.jsonl      educational web text (technical pages + a sample of the rest)
  chat.jsonl         chat conversations formatted with the chat template
"""
import glob
import json
import os
import random
import re

import pyarrow.parquet as pq

from common import CORPUS, RAW, format_chat

random.seed(1234)

TECH_CATS = {"COMPUTERS", "TECHNOLOGY & ENGINEERING", "SCIENCE", "MATHEMATICS"}
TECH_WORDS = re.compile(
    r"\b(computer|computing|software|hardware|network|networking|internet|router|protocol|"
    r"server|database|algorithm|programming|program|code|processor|cpu|memory|ram|"
    r"operating system|linux|windows|wifi|wi-fi|ethernet|tcp|udp|http|dns|ip address|"
    r"encryption|cyber|security|data|digital|electronic|circuit|byte|bit|binary|website|"
    r"browser|cloud|storage|engineer|technology|machine learning|python|java)\b", re.I)

# keep-probability for non-technical cosmopedia docs, by audience (simpler English preferred)
AUDIENCE_KEEP = {
    "young_children": 0.55, "children": 0.55, "middle_school_students": 0.45,
    "general": 0.45, "high_school_studnets": 0.35, "college_students": 0.2,
    "college_studnets": 0.2, "researchers": 0.05, "requires_details": 0.05, "alien": 0.05,
}


def clean(text):
    text = text.replace("\r\n", "\n").replace(" ", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tech_density(text):
    return len(TECH_WORDS.findall(text)) / max(1, len(text) / 1000)


def do_cosmopedia(raw=RAW, out_name="cosmopedia.jsonl"):
    n_tech = n_other = tok = 0
    with open(os.path.join(CORPUS, out_name), "w") as out:
        for f in sorted(glob.glob(os.path.join(raw, "smollm-corpus/cosmopedia-v2/*.parquet"))):
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=4096, columns=["prompt", "text", "audience", "token_length"]):
                d = batch.to_pydict()
                for prompt, text, aud, tl in zip(d["prompt"], d["text"], d["audience"], d["token_length"]):
                    m = re.search(r'as part of "([^"]+)"', prompt)
                    cat = m.group(1).split(" / ")[0] if m else ""
                    is_tech = cat in TECH_CATS or tech_density(text) > 6
                    if is_tech:
                        keep = True
                        n_tech += 1
                    else:
                        keep = random.random() < AUDIENCE_KEEP.get(aud, 0.2)
                        n_other += keep
                    if keep:
                        tok += tl
                        out.write(json.dumps({"text": clean(text), "tech": is_tech}) + "\n")
            print(f, "tech", n_tech, "other", n_other, "approx tokens %.0fM" % (tok / 1e6), flush=True)


def do_fineweb(raw=RAW, out_name="fineweb.jsonl"):
    n_tech = n_other = tok = 0
    with open(os.path.join(CORPUS, out_name), "w") as out:
        for f in sorted(glob.glob(os.path.join(raw, "smollm-corpus/fineweb-edu-dedup/*.parquet"))):
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=4096, columns=["text", "metadata"]):
                d = batch.to_pydict()
                for text, meta in zip(d["text"], d["metadata"]):
                    tc = meta["token_count"]
                    if tc < 100 or tc > 3000:
                        continue
                    dens = tech_density(text)
                    if dens > 8:
                        keep, is_tech = True, True
                        n_tech += 1
                    else:
                        is_tech = False
                        keep = meta["int_score"] >= 3 and random.random() < 0.06
                        n_other += keep
                    if keep:
                        tok += tc
                        out.write(json.dumps({"text": clean(text), "tech": is_tech}) + "\n")
            print(f, "tech", n_tech, "other", n_other, "approx tokens %.0fM" % (tok / 1e6), flush=True)


def norm_messages(msgs):
    out = []
    for m in msgs:
        if m["role"] == "system":
            continue
        if out and out[-1]["role"] == m["role"]:
            return None
        out.append({"role": m["role"], "content": m["content"]})
    if not out or out[0]["role"] != "user" or out[-1]["role"] != "assistant":
        return None
    return out


def do_chat(raw=RAW, out_name="chat.jsonl", max_chars=2200):
    n = 0
    with open(os.path.join(CORPUS, out_name), "w") as out:
        files = sorted(glob.glob(os.path.join(raw, "smol-smoltalk/data/train-*.parquet")))
        files += sorted(glob.glob(os.path.join(raw, "smoltalk/data/everyday-conversations/train-*.parquet")))
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=4096, columns=["messages"]):
                for msgs in batch.to_pydict()["messages"]:
                    msgs = norm_messages(msgs)
                    if msgs is None:
                        continue
                    # keep the first exchanges that fit comfortably in the context window
                    while msgs and len(format_chat(msgs)) > max_chars:
                        msgs = msgs[:-2]
                    if not msgs:
                        continue
                    text = format_chat(msgs)
                    if "```" in text and random.random() < 0.7:
                        continue  # the model is meant to talk, not write code
                    out.write(json.dumps({"messages": msgs}) + "\n")
                    n += 1
            print(f, "conversations", n, flush=True)


if __name__ == "__main__":
    import sys
    if "--gorilla" in sys.argv:
        # Extra shards for Probabilistic Gorilla (downloaded by download_data.py --gorilla). Chats may be
        # longer because Gorilla has a 1024-token context.
        RAW_GORILLA = RAW + "_gorilla"
        do_chat(RAW_GORILLA, "chat_gorilla.jsonl", max_chars=4400)
        do_cosmopedia(RAW_GORILLA, "cosmopedia_gorilla.jsonl")
        do_fineweb(RAW_GORILLA, "fineweb_gorilla.jsonl")
    else:
        do_chat()
        do_cosmopedia()
        do_fineweb()
