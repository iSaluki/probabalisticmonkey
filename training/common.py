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

for d in (WORK, RAW, CORPUS, TOKENS, CKPT):
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
