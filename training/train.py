"""Step 5/6: train Probabilistic Monkey.

  python train.py pretrain     # next-token prediction on the mixed corpus
  python train.py sft          # chat fine-tuning (loss only on assistant replies)

Training is resumable: rerunning the same command continues from the latest
checkpoint in $PM_WORK/checkpoints/<phase>/.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

import model as M
from common import CKPT, MODEL_DIR, TOKENS, format_chat
from model import Config, Monkey

SEQ = 512


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["pretrain", "sft"])
    p.add_argument("--micro_batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min_lr_frac", type=float, default=0.05)
    p.add_argument("--total_steps", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--decay_frac", type=float, default=0.25, help="WSD: final fraction of steps spent decaying")
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=250)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--no_compile", action="store_true")
    a = p.parse_args()
    if a.phase == "pretrain":
        a.lr = a.lr or 1.5e-3
        a.total_steps = a.total_steps or 30000
        a.warmup = a.warmup if a.warmup is not None else 500
    else:
        a.lr = a.lr or 3e-4
        a.warmup = a.warmup if a.warmup is not None else 50
    return a


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    if a.phase == "pretrain":  # warmup-stable-decay
        decay_start = int(a.total_steps * (1 - a.decay_frac))
        if step < decay_start:
            return a.lr
        frac = min(1.0, (step - decay_start) / max(1, a.total_steps - decay_start))
        return a.lr * (1 - frac * (1 - a.min_lr_frac))
    frac = min(1.0, (step - a.warmup) / max(1, a.total_steps - a.warmup))  # cosine
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac)))


class PretrainData:
    def __init__(self, B):
        self.data = np.memmap(os.path.join(TOKENS, "pretrain.bin"), dtype=np.uint16, mode="r")
        self.n_chunks = (len(self.data) - 1) // SEQ
        self.perm = np.random.default_rng(0).permutation(self.n_chunks)
        self.B = B
        self.val = np.memmap(os.path.join(TOKENS, "val.bin"), dtype=np.uint16, mode="r")

    def batch(self, i):
        idx = self.perm[(np.arange(i * self.B, (i + 1) * self.B)) % self.n_chunks]
        x = np.stack([self.data[j * SEQ: j * SEQ + SEQ + 1] for j in idx]).astype(np.int64)
        x = torch.from_numpy(x)
        return x[:, :-1], x[:, 1:], None

    def val_batches(self, n=16):
        out = []
        for k in range(n):
            x = np.stack([self.val[(k * 8 + j) * SEQ:(k * 8 + j) * SEQ + SEQ + 1] for j in range(8)]).astype(np.int64)
            x = torch.from_numpy(x)
            out.append((x[:, :-1], x[:, 1:], None))
        return out


class SFTData:
    def __init__(self, B):
        rows = np.fromfile(os.path.join(TOKENS, "sft.bin"), dtype=np.uint16).reshape(-1, SEQ + 1)
        masks = np.fromfile(os.path.join(TOKENS, "sft_mask.bin"), dtype=np.uint8).reshape(-1, SEQ + 1)
        perm = np.random.default_rng(0).permutation(len(rows))
        rows, masks = rows[perm], masks[perm]
        self.vrows, self.vmasks = rows[:128], masks[:128]
        self.rows, self.masks = rows[128:], masks[128:]
        self.B = B
        self.n_chunks = len(self.rows)

    def batch(self, i):
        idx = np.arange(i * self.B, (i + 1) * self.B) % len(self.rows)
        x = torch.from_numpy(self.rows[idx].astype(np.int64))
        m = torch.from_numpy(self.masks[idx].astype(np.int64))
        return x[:, :-1], x[:, 1:], m[:, 1:]

    def val_batches(self):
        out = []
        for k in range(0, 128, 8):
            x = torch.from_numpy(self.vrows[k:k + 8].astype(np.int64))
            m = torch.from_numpy(self.vmasks[k:k + 8].astype(np.int64))
            out.append((x[:, :-1], x[:, 1:], m[:, 1:]))
        return out


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    losses = []
    for x, y, m in data.val_batches():
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, loss, _ = model(x, y, m)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample(model, tok, prompt, n=60):
    model.eval()
    ids = tok.encode(prompt).ids
    end_ids = {tok.token_to_id("<|end|>"), tok.token_to_id("<|endoftext|>")}
    for _ in range(n):
        x = torch.tensor([ids[-SEQ:]])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(x)[0, -1].float()
        probs = torch.softmax(logits / 0.7, -1)
        nxt = int(torch.multinomial(probs, 1))
        if nxt in end_ids:
            break
        ids.append(nxt)
    model.train()
    return tok.decode(ids[len(tok.encode(prompt).ids):])


def save(path, model, opt, step, args):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "config": model.cfg.to_dict(), "args": vars(args)}, tmp)
    os.replace(tmp, path)


def main():
    a = get_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    cfg = Config(vocab_size=tok.get_vocab_size(), d_model=512, n_layers=10, n_heads=8,
                 n_experts=8, d_expert=1024, d_shared=512, max_seq_len=SEQ)
    if not a.no_compile:
        M.enable_compile()
    model = Monkey(cfg)
    total, active = model.param_counts()
    print(f"params: total {total/1e6:.1f}M, active per token {active/1e6:.1f}M", flush=True)

    ckdir = os.path.join(CKPT, a.phase)
    os.makedirs(ckdir, exist_ok=True)
    latest = os.path.join(ckdir, "latest.pt")
    B = a.micro_batch * a.accum
    data = PretrainData(B) if a.phase == "pretrain" else SFTData(B)
    if a.phase == "sft" and a.total_steps is None:
        a.total_steps = 2 * data.n_chunks // B  # two epochs

    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (decay if p.dim() >= 2 and "embed" not in n_ else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    step = 0
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print("resumed from step", step, flush=True)
    elif a.phase == "sft":
        ck = torch.load(os.path.join(CKPT, "pretrain", "final.pt"), map_location="cpu")
        model.load_state_dict(ck["model"])
        print("initialised from pretrained model", flush=True)

    log = open(os.path.join(ckdir, "log.jsonl"), "a")
    print(f"phase {a.phase}: {a.total_steps} steps x {B*SEQ} tokens = {a.total_steps*B*SEQ/1e6:.0f}M tokens", flush=True)
    model.train()
    t0 = time.time()
    tok_count = 0
    while step < a.total_steps:
        lr = lr_at(step, a)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = aux_acc = 0.0
        for k in range(a.accum):
            x, y, m = data.batch(step * a.accum + k)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                _, loss, aux = model(x, y, m)
            ((loss + cfg.aux_loss_coef * aux) / a.accum).backward()
            loss_acc += loss.item() / a.accum
            aux_acc += aux.item() / a.accum
            tok_count += x.numel()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            el = time.time() - t0
            rec = {"step": step, "loss": round(loss_acc, 4), "aux": round(aux_acc, 4), "lr": lr,
                   "gnorm": round(float(gn), 3), "tok_s": round(tok_count / el)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            t0, tok_count = time.time(), 0
        if step % a.ckpt_every == 0 or step == a.total_steps:
            save(latest, model, opt, step, a)
        if step % a.eval_every == 0 or step == a.total_steps:
            vl = evaluate(model, data)
            prompt = format_chat([{"role": "user", "content": "What does a router do?"}], add_generation_prompt=True)
            s1 = sample(model, tok, prompt)
            s2 = sample(model, tok, "The internet is")
            rec = {"step": step, "val_loss": round(vl, 4), "sample_chat": s1, "sample_text": s2}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            t0, tok_count = time.time(), 0
    save(os.path.join(ckdir, "final.pt"), model, opt, step, a)
    print("done", flush=True)


if __name__ == "__main__":
    main()
