"""Step 3: train a byte-level BPE tokenizer (16k vocabulary) on a sample of the corpus."""
import json
import os
import random

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from common import CORPUS, MODEL_DIR, SPECIAL_TOKENS, format_chat
from handwritten import load_notes, qa_conversations

VOCAB_SIZE = 16384


def sample_texts(max_chars=150_000_000):
    rng = random.Random(0)
    budget = {"cosmopedia.jsonl": 0.5, "fineweb.jsonl": 0.35, "chat.jsonl": 0.15}
    for fname, frac in budget.items():
        n = 0
        with open(os.path.join(CORPUS, fname)) as f:
            for line in f:
                if rng.random() > 0.3:
                    continue
                d = json.loads(line)
                t = format_chat(d["messages"]) if "messages" in d else d["text"]
                n += len(t)
                yield t
                if n > max_chars * frac:
                    break
    for _ in range(20):
        yield from load_notes()
    for conv in qa_conversations(5):
        yield format_chat(conv)


if __name__ == "__main__":
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=False,
    )
    tok.train_from_iterator(sample_texts(), trainer=trainer)
    os.makedirs(MODEL_DIR, exist_ok=True)
    out = os.path.join(MODEL_DIR, "tokenizer.json")
    tok.save(out)
    print("saved", out, "vocab", tok.get_vocab_size())
    for s in ["Hello! How does a router forward packets?", "<|user|>\nhi<|end|>\n<|assistant|>\n"]:
        print(s, "->", tok.encode(s).tokens)
