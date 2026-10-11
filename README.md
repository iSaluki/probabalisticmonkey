# Probabilistic Monkey

Probabilistic Monkey is a small chat language model with **152.6 million parameters**. It is trained on English text, with a skew towards technical topics, and runs at chatting speed on an ordinary computer's CPU. No GPU is needed. This repository contains the whole pipeline (data, tokenizer, training, export and evaluation), the trained model, and a deliberately bare terminal-style web page for talking to it.

The model, code, data selection, hand-written teaching material and benchmark were all produced, tested and verified by Claude. See [the original brief](#the-original-brief) at the end.

## Quick start

You need Python 3.9 or newer and about 1 GB of free RAM.

```bash
git clone https://github.com/iSaluki/probabalisticmonkey.git
cd probabalisticmonkey
./run.sh          # Linux / macOS
run.bat           # Windows (double-click works too)
```

On the first run the launcher creates a virtual environment in `.venv` and installs the two small dependencies, `numpy` and `tokenizers`. It then starts a local server at <http://127.0.0.1:8642> and opens your browser. Type a message and press **Enter**. The whole conversation so far is sent with each message, so the monkey can refer back to earlier turns. There are no other buttons or features. Press Ctrl+C in the terminal to stop.

Options: `./run.sh --port 9000`, `./run.sh --no-browser`.

## What it can and can't do

It speaks simple, grammatical English and answers basic questions about computers and networks correctly. For example, this exchange is from the real web page:

```
> What does DNS do?
DNS, the Domain Name System, translates domain names like example.com into IP addresses,
so your computer knows where to connect. It works like a phone book for the internet.

> Thanks! And what port does HTTPS use?
HTTP uses port 80 and HTTPS uses port 443. SSH uses port 22 and DNS uses port 53.
```

It is a very small model trained in a few days on a 4-core CPU, so expect it to be confidently wrong at times, to drift off-topic in long answers, and to know little outside common general knowledge and basic computing and networking. It cannot browse the internet.

## The model

| | |
|---|---|
| Architecture | decoder-only transformer, sparse **mixture of experts** |
| Parameters | **152.6M total**, 42.5M active per token |
| Layers / width / heads | 10 / 512 / 8 (head size 64), RoPE positions, RMSNorm, SwiGLU |
| Feed-forward | per layer: 1 always-on shared expert (512 hidden) + 8 routed experts (1024 hidden), top-1 routing |
| Context window | 512 tokens (older turns are dropped when a chat gets longer) |
| Tokenizer | byte-level BPE, 16,384 tokens, trained on the corpus |
| Shipped weights | int8 with per-channel scales, ~150 MB in four `model/weights-*.npz` shards |

**Why mixture of experts:** the brief asks for at least 100M parameters *and* speed on consumer hardware. Each token passes through the shared expert and only one of the eight routed experts, so the model has 152.6M parameters of capacity but does the work of a ~42M-parameter model per token. That made training on a CPU feasible, and it makes chatting fast.

**Why NumPy for inference:** `app/engine.py` is a dependency-light reimplementation of the forward pass with a KV cache. It needs neither PyTorch nor a GPU. I checked it against the PyTorch training model: the logits correlate at 0.9998, and the small gap comes from int8 quantisation. It generates roughly 60–70 tokens per second on one CPU core.

## How it was trained

Everything ran on a 4-core Intel Xeon CPU (AVX-512 / AMX bf16) with 15 GB of RAM, in bfloat16 autocast with `torch.compile`.

1. **Data** (`training/download_data.py`, `training/prepare_corpus.py`). Public, permissively licensed sources from the Hugging Face Hub:
   * [Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) synthetic textbooks and stories. All computing, technology, science and maths documents are kept, plus a sample of the rest that prefers simpler audiences.
   * [FineWeb-Edu (dedup)](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) educational web pages. Technically dense pages are kept, plus a sample of other high-quality pages.
   * [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) and [everyday-conversations](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) chats. Conversations are shortened to fit the context window, and most code-heavy ones are dropped.
   * Hand-written teaching material by Claude in `training/data/`: a short textbook on how computers and networks work, and about 210 question/answer facts with alternative phrasings (`tech_qa.txt` gives full-sentence answers, `tech_short.txt` gives short ones).

   Technical documents are included twice, which skews the 547M-token pretraining mix towards technical topics.
2. **Tokenizer** (`training/train_tokenizer.py`): byte-level BPE with chat tokens `<|user|>`, `<|assistant|>`, `<|end|>` and `<|endoftext|>`.
3. **Pretraining** (`training/train.py pretrain`): 17,000 steps × 32,768 tokens, about 557M tokens or one pass over the corpus. AdamW with lr 1.2e-3 and a warmup-stable-decay schedule. A Switch-Transformer load-balancing loss keeps all experts in use. This took about 2.5 days, and the validation loss fell from 9.3 to 2.21.
4. **Chat fine-tuning** (`training/train.py sft`): one epoch over about 90k conversations plus the hand-written Q&A. The loss is applied only to the assistant's replies.
5. **Export** (`training/export.py`): int8 quantisation into `model/`.
6. **Evaluation** (`eval/benchmark.py`): see below.

To reproduce from scratch (needs about 15 GB of disk and several days on a similar CPU):

```bash
pip install -r training/requirements.txt
cd training
python download_data.py && python prepare_corpus.py && python train_tokenizer.py && python tokenize_data.py
python train.py pretrain && python train.py sft && python export.py
cd .. && python eval/benchmark.py
```

Intermediate files go to `../pm_data` (set `PM_WORK` to change it). Training is resumable: rerun the same command after an interruption.

## Benchmark

The brief sets two acceptance criteria: the model must speak comprehensible English, and it must pass a very simple test on how computers and networks work. `eval/questions.json` holds the test, which was written before training:

* **20 multiple-choice questions** on computers and networks. Each option is scored by the average log-probability the model assigns to it as its answer. The pass mark is 75%.
* **15 open-ended questions.** The model answers in its own words (greedy decoding), and an answer passes if it mentions a key concept. The pass mark is 60%.
* **5 everyday English prompts** whose replies are reviewed by reading them.

The test questions were written separately from the teaching material, with different wording. `eval/check_leakage.py` checks that none of them appears word for word in the hand-written training data. The facts themselves *are* taught. The test measures whether the model learned them, not whether it can guess them.

### Results

| Test | Score | Pass mark | |
|---|---|---|---|
| Multiple choice (computers & networks) | **17/20 (85%)** | 75% | pass |
| Open-ended (computers & networks) | **14/15 (93%)** | 60% | pass |
| English | fluent, grammatical, on topic | read by Claude | pass |

**Verdict: the model meets both acceptance criteria.**

The three multiple-choice misses: it picked *a hard disk* rather than *RAM* as the memory that is wiped at power-off, *a webcam* rather than *a router* for forwarding packets between networks, and *the server* rather than *the client* for the role of your web browser. When asked the same things in open questions it answers correctly, so these are mostly weaknesses of the scoring method on a small model. The open-ended miss is a genuine mistake: it described a byte as "a small piece of data sent over a network".

The English replies are clear and well formed, but the model is weak at keeping a story consistent. In the dog story the dog turns into a cat halfway through.

### How the benchmark was passed (full disclosure)

* **First attempt (failed).** The first fine-tuned model scored 15/15 on the open-ended questions but only **12/20** on multiple choice, so it failed. It knew the facts but had only ever been taught to answer in full paragraphs, so it gave bare options like "The CPU" low probability.
* **A rejected change to the test.** I briefly tried showing the options inside the question. That scored 6/20, because the model just copied the first or last option, so I reverted it. The test is exactly as written before training.
* **Fix.** I added `training/data/tech_short.txt`, about 110 short-answer questions covering every fact in the teaching material plus some general knowledge, and re-ran chat fine-tuning. Pretraining was not repeated. The result is the 17/20 above.
* **Leakage.** `eval/check_leakage.py` finds none of the benchmark questions in the teaching data. It also flags shared 6-word phrases. Two benchmark questions share one with the short textbook (`tech_notes.md`), which was written before the benchmark and used in pretraining:
  * The RAM question shares "when the computer is switched off". The textbook uses that phrase about *storage*, so it gives nothing away.
  * The video-call question shares "speed matters more than perfect reliability". The textbook says UDP is used where speed matters more than perfect reliability, which is the fact that question tests.

The full report, with every answer, is in [`eval/results.md`](eval/results.md).

## Probabilistic Monkey Big (in training)

A larger sibling with **516.9M parameters**, of which 74.5M are active per token. It uses the same design, tokenizer and app, and is being trained for a harder job: a technical helper that can troubleshoot simple problems, explain processes step by step, do basic technical calculations and keep track of a conversation.

| | Probabilistic Monkey | Big |
|---|---|---|
| Parameters (total / active per token) | 152.6M / 42.5M | 516.9M / 74.5M |
| Layers / width / heads | 10 / 512 / 8 | 12 / 640 / 10 |
| Experts per layer | 1 shared + 8 routed | 1 shared + 16 routed |
| Context window | 512 tokens | 1024 tokens |
| Pretraining data | ~557M tokens | ~790M tokens, including extra Cosmopedia, FineWeb-Edu and chat shards |
| Optimiser state | fp32 | bf16 (so it fits in 15 GB of RAM) |

**The harder task.** [`eval/big_questions.json`](eval/big_questions.json) was written before any Big training data. The pass marks are:

| Section | What it tests | Pass mark |
|---|---|---|
| Multiple choice (15) | harder computing and networking facts | 80% |
| Troubleshooting (8) | for example "I can ping 8.8.8.8 but websites won't load" → DNS | 60% |
| Step-by-step explanations (3) | loading a web page, booting a computer, joining Wi-Fi | 2 of 3 |
| Calculations (10) | bits and bytes, binary, download times, subnet sizes | 50% |
| Conversation memory (5) | recalling a fact the user gave earlier in the chat | 60% |
| English | longer writing: a story, an email, an explanation | read by Claude |

For comparison, the small model scores 8/15, 2/8, 2/3, 1/10 and 1/5 on this test, so it fails.

**New teaching data.**
* `training/data/tech_processes.txt`: hand-written step-by-step explanations.
* `training/synthetic.py`: generated practice conversations for calculations (with short worked answers), troubleshooting and conversation memory. The values used in the benchmark are excluded on purpose (for example it never sees a /29 subnet), so the model has to generalise. `eval/check_leakage.py` also checks these exact generated conversations against the benchmark.

**Reproduce:**

```bash
cd training
python download_data.py --big && python prepare_corpus.py --big && python tokenize_data.py big
python train.py pretrain --variant big --lr 1e-3 --total_steps 24000
python train.py sft --variant big && python export.py --variant big
cd .. && python eval/big_benchmark.py
```

On the 4-core CPU used here, pretraining runs at about 1,350 tokens/s, so it takes about a week. When it is finished, `./run.sh --big` will chat with it.

## Deploying to Cloudflare

A plain Cloudflare Worker cannot hold either model: an isolate is limited to 128 MB of memory, and the weights alone are 150 MB and 520 MB. [`deploy/cloudflare/`](deploy/cloudflare/) instead deploys a Worker in front of a [Cloudflare Container](https://developers.cloudflare.com/containers/) that runs the same server and model. It needs the Workers Paid plan and Docker; then:

```bash
cd deploy/cloudflare && npm install && npx wrangler login && npm run deploy
```

See [`deploy/cloudflare/README.md`](deploy/cloudflare/README.md) for the reasoning, the configuration, the cost and what was tested.

## Repository layout

```
run.sh / run.bat        one-file launchers (add --big for the Big model)
app/                    web server, terminal page and NumPy inference engine
model/                  trained weights (int8), config and tokenizer
model_big/              Probabilistic Monkey Big (once trained)
training/               the full training pipeline and hand-written teaching data
eval/                   benchmarks, runners, leakage check and results
deploy/cloudflare/      Cloudflare Workers + Containers deployment
```

## The original brief

> Probabilistic Monkey is an LLM. It has at least 100M parameters and is trained on English language, with a skew towards technical areas.
>
> It is engineered to be efficient and capable of running on consumer hardware at a good speed. This repository should hold an entire pipeline for training the model, the model outputs and an exceptionally basic web interface for interacting with it.
>
> If a user clone the repo, they should be able to run a single .sh or .bat file to launch the model in a local web session. This web session should be super simple, with a text input and output in a terminal themed page with no other buttons or features. Pressing enter sends the current input and conversation history to generate an output.
>
> Probablistic Monkey should be entirely produced, tested and verified by Claude. No human oversight will be offered beyond this initial prompt.
>
> The benchmark for the model being acceptable is: must be able to speak in comprehensible English. Must be able to pass a very simple test on how computers and networks work.
