"""Step 7: export a trained checkpoint to the compact int8 format used by app/engine.py."""
import argparse
import json
import os
import shutil

import numpy as np
import torch

from common import MODEL_DIR, VARIANTS

SHARD_BYTES = 45_000_000  # keep every file well under GitHub's 100 MB limit


def quantize(t, axis):
    """Symmetric int8 quantisation with one scale per slice along `axis` (the output channel)."""
    t = t.float().numpy()
    red = tuple(i for i in range(t.ndim) if i != axis)
    scale = np.abs(t).max(axis=red, keepdims=True) / 127.0
    scale = np.maximum(scale, 1e-12)
    q = np.clip(np.round(t / scale), -127, 127).astype(np.int8)
    return q, scale.astype(np.float16)


def convert(state, cfg):
    out = {}
    for k, v in state.items():
        k = k.replace(".weight", "").replace("attn.", "").replace("moe.", "")
        if v.dim() == 1:
            out[k] = v.float().numpy().astype(np.float32)
        elif k.endswith(("w_in", "w_out")):  # [E, in, out] -> one tensor per expert, scale per output column
            for e in range(v.shape[0]):
                q, s = quantize(v[e], axis=1)
                out[f"{k}.{e}"], out[f"{k}.{e}.scale"] = q, s
        elif k.endswith("router"):
            out[k] = v.float().numpy().astype(np.float16)
        else:  # nn.Linear / embedding: [out, in], scale per row
            q, s = quantize(v, axis=0)
            out[k], out[k + ".scale"] = q, s
    return out


def write(arrays, cfg, model_dir):
    for f in os.listdir(model_dir):
        if f.startswith("weights-"):
            os.remove(os.path.join(model_dir, f))
    shards, cur, size = [], {}, 0
    for k in arrays:
        cur[k] = arrays[k]
        size += arrays[k].nbytes
        if size > SHARD_BYTES:
            shards.append(cur)
            cur, size = {}, 0
    if cur:
        shards.append(cur)
    for i, s in enumerate(shards):
        np.savez(os.path.join(model_dir, f"weights-{i:02d}.npz"), **s)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {len(shards)} shards to {model_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="small")
    ap.add_argument("--ckpt", default=None, help="default: the variant's final chat checkpoint")
    ap.add_argument("--out", default=None, help="default: the variant's model directory")
    a = ap.parse_args()
    v = VARIANTS[a.variant]
    a.ckpt = a.ckpt or os.path.join(v["ckpt"], "sft", "final.pt")
    a.out = a.out or v["model_dir"]
    ck = torch.load(a.ckpt, map_location="cpu")
    os.makedirs(a.out, exist_ok=True)
    write(convert(ck["model"], ck["config"]), ck["config"], a.out)
    if os.path.abspath(a.out) != os.path.abspath(MODEL_DIR):  # all variants share one tokenizer
        shutil.copy(os.path.join(MODEL_DIR, "tokenizer.json"), os.path.join(a.out, "tokenizer.json"))
