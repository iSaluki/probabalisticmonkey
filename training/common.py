"""Shared paths and constants for the training pipeline."""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Large intermediate files live outside the repository (override with PM_WORK).
WORK = os.environ.get("PM_WORK", os.path.join(os.path.dirname(REPO), "pm_data"))
RAW = os.path.join(WORK, "raw")
CORPUS = os.path.join(WORK, "corpus")
TOKENS = os.path.join(WORK, "tokens")
CKPT = os.path.join(WORK, "checkpoints")
MODEL_DIR = os.path.join(REPO, "model")
HANDWRITTEN = os.path.join(REPO, "training", "data")

# Chat template special tokens
EOT = "<|endoftext|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"
SPECIAL_TOKENS = [EOT, USER, ASSISTANT, END]

# Model variants. Both share the tokenizer in model/tokenizer.json.
VARIANTS = {
    # Probabilistic Monkey: 152.6M parameters, 42.5M active per token
    "small": {
        "model": dict(d_model=512, n_layers=10, n_heads=8, n_experts=8, d_expert=1024, d_shared=512, max_seq_len=512),
        "tokens": TOKENS, "ckpt": CKPT, "model_dir": MODEL_DIR,
        "micro_batch": 32, "accum": 2, "bf16_optim": False,
    },
    # Probabilistic Gorilla: 516.9M parameters, 74.5M active per token, 1024-token context
    "gorilla": {
        "model": dict(d_model=640, n_layers=12, n_heads=10, n_experts=16, d_expert=1280, d_shared=640, max_seq_len=1024),
        "tokens": os.path.join(WORK, "tokens_gorilla"), "ckpt": os.path.join(WORK, "checkpoints_gorilla"),
        "model_dir": os.path.join(REPO, "model_gorilla"),
        "micro_batch": 8, "accum": 4, "bf16_optim": True,
    },
}

for d in (WORK, RAW, CORPUS, TOKENS, CKPT, VARIANTS["gorilla"]["tokens"], VARIANTS["gorilla"]["ckpt"]):
    os.makedirs(d, exist_ok=True)


def format_chat(messages, add_generation_prompt=False):
    """messages: list of {"role": "user"|"assistant", "content": str}."""
    out = []
    for m in messages:
        tag = USER if m["role"] == "user" else ASSISTANT
        out.append(f"{tag}{m['content'].strip()}{END}")
    if add_generation_prompt:
        out.append(ASSISTANT)
    return "".join(out)
