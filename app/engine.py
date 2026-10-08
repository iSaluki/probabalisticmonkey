"""Pure-NumPy inference engine for Probabilistic Monkey.

Loads the int8-quantised weights in model/ and generates text with a KV cache.
Only numpy and tokenizers are needed - no deep learning framework.
"""
import glob
import json
import os

import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(HERE), "model")

USER, ASSISTANT, END, EOT = "<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>"


def load_weights(model_dir):
    """Read every weights-*.npz shard and de-quantise int8 tensors to float32."""
    raw = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "weights-*.npz"))):
        with np.load(f) as z:
            raw.update({k: z[k] for k in z.files})
    w = {}
    for k, v in raw.items():
        if k.endswith(".scale"):
            continue
        if v.dtype == np.int8:
            w[k] = v.astype(np.float32) * raw[k + ".scale"].astype(np.float32)
        else:
            w[k] = v.astype(np.float32)
    return w


def rms_norm(x, weight, eps=1e-5):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def silu(x):
    return x / (1.0 + np.exp(-x))


class Monkey:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR):
        with open(os.path.join(model_dir, "config.json")) as f:
            self.cfg = json.load(f)
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.w = load_weights(model_dir)
        c = self.cfg
        self.L, self.H, self.D = c["n_layers"], c["n_heads"], c["d_model"]
        self.hd = self.D // self.H
        self.max_len = c["max_seq_len"]
        inv = 1.0 / (c["rope_theta"] ** (np.arange(0, self.hd, 2, dtype=np.float64) / self.hd))
        freqs = np.outer(np.arange(self.max_len, dtype=np.float64), inv)
        self.cos = np.cos(freqs).astype(np.float32)
        self.sin = np.sin(freqs).astype(np.float32)
        self.ids = {t: self.tok.token_to_id(t) for t in (USER, ASSISTANT, END, EOT)}

    # ---- model maths -------------------------------------------------
    def _rope(self, x, start):
        # x: H, T, hd
        T = x.shape[1]
        cos, sin = self.cos[start:start + T], self.sin[start:start + T]
        x1, x2 = x[..., : self.hd // 2], x[..., self.hd // 2:]
        return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

    def _moe(self, x, p):
        w = self.w
        g, u = np.split(x @ w[p + "shared_in"].T, 2, axis=-1)
        out = (silu(g) * u) @ w[p + "shared_out"].T
        logits = x @ w[p + "router"].T
        logits = logits - logits.max(-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(-1, keepdims=True)
        top = probs.argmax(-1)
        for e in np.unique(top):
            sel = np.nonzero(top == e)[0]
            ge, ue = np.split(x[sel] @ w[p + f"w_in.{e}"], 2, axis=-1)
            out[sel] += ((silu(ge) * ue) @ w[p + f"w_out.{e}"]) * probs[sel, e][:, None]
        return out

    def forward(self, ids, cache):
        """Run new tokens `ids` through the model; cache holds past keys/values."""
        w = self.w
        start = cache["len"]
        T = len(ids)
        x = w["embed"][ids]
        for l in range(self.L):
            p = f"blocks.{l}."
            h = rms_norm(x, w[p + "n1"])
            qkv = (h @ w[p + "qkv"].T).reshape(T, 3, self.H, self.hd).transpose(1, 2, 0, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = self._rope(q, start), self._rope(k, start)
            if cache["k"][l] is not None:
                k = np.concatenate([cache["k"][l], k], axis=1)
                v = np.concatenate([cache["v"][l], v], axis=1)
            cache["k"][l], cache["v"][l] = k, v
            att = (q @ k.transpose(0, 2, 1)) / np.sqrt(self.hd)
            S = k.shape[1]
            mask = np.arange(S)[None, :] > (np.arange(T)[:, None] + start)
            att = np.where(mask[None], -1e30, att)
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            y = (att @ v).transpose(1, 0, 2).reshape(T, self.D)
            x = x + y @ w[p + "proj"].T
            x = x + self._moe(rms_norm(x, w[p + "n2"]), p)
        cache["len"] += T
        x = rms_norm(x[-1:], w["norm"])
        return (x @ w["embed"].T)[0]

    def new_cache(self):
        return {"len": 0, "k": [None] * self.L, "v": [None] * self.L}

    # ---- prompting ---------------------------------------------------
    def encode_chat(self, messages, reserve):
        """Token ids for a chat; drops the oldest turns so `reserve` tokens fit for the reply."""
        budget = self.max_len - reserve
        turns = []
        for m in messages:
            tag = USER if m["role"] == "user" else ASSISTANT
            turns.append(self.tok.encode(f"{tag}{m['content'].strip()}{END}").ids)
        tail = [self.ids[ASSISTANT]]
        kept = []
        for t in reversed(turns):
            if sum(map(len, kept)) + len(t) + len(tail) > budget:
                if not kept:  # a single very long message: keep its end
                    kept.append(t[-(budget - len(tail)):])
                break
            kept.append(t)
        ids = [i for t in reversed(kept) for i in t] + tail
        return ids

    def generate(self, messages, max_new_tokens=200, temperature=0.7, top_k=40, top_p=0.9,
                 repetition_penalty=1.15, seed=None):
        """Yields the reply text piece by piece."""
        rng = np.random.default_rng(seed)
        max_new_tokens = min(max_new_tokens, self.max_len // 2)
        ids = self.encode_chat(messages, reserve=max_new_tokens)
        cache = self.new_cache()
        logits = self.forward(ids, cache)
        out, printed = [], ""
        stop = {self.ids[END], self.ids[EOT], self.ids[USER]}
        for _ in range(max_new_tokens):
            logits = logits.astype(np.float64)
            if repetition_penalty != 1.0 and out:
                seen = np.unique(out[-64:])
                logits[seen] = np.where(logits[seen] > 0, logits[seen] / repetition_penalty,
                                        logits[seen] * repetition_penalty)
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                logits = logits / temperature
                if top_k:
                    kth = np.partition(logits, -top_k)[-top_k]
                    logits = np.where(logits < kth, -np.inf, logits)
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                order = np.argsort(-probs)
                cum = np.cumsum(probs[order])
                cut = order[np.searchsorted(cum, top_p) + 1:]
                probs[cut] = 0
                probs /= probs.sum()
                nxt = int(rng.choice(len(probs), p=probs))
            if nxt in stop:
                break
            out.append(nxt)
            text = self.tok.decode(out)
            if not text.endswith("�"):  # wait for complete multi-byte characters
                yield text[len(printed):]
                printed = text
            if cache["len"] >= self.max_len:
                break
            logits = self.forward([nxt], cache)

    def reply(self, messages, **kw):
        return "".join(self.generate(messages, **kw)).strip()

    def score(self, messages, answer):
        """Average log-probability of `answer` as the assistant's reply (for multiple choice)."""
        prompt = self.encode_chat(messages, reserve=64)
        ans = self.tok.encode(f"{answer}{END}").ids
        cache = self.new_cache()
        total = 0.0
        logits = self.forward(prompt, cache)
        for t in ans:
            l = logits.astype(np.float64)
            l = l - l.max()
            total += l[t] - np.log(np.exp(l).sum())
            logits = self.forward([t], cache)
        return total / len(ans)
