# Deploying Probabilistic Monkey on Cloudflare

This folder deploys the chat page and the model to Cloudflare as a **Worker in front of a Cloudflare Container**. The Worker receives every request and forwards it to a container. The container runs the same NumPy server and model as `./run.sh` and goes to sleep when nobody is using it.

## Why not a plain Worker?

A Worker on its own cannot run these models. Cloudflare's [Workers limits](https://developers.cloudflare.com/workers/platform/limits/) are the problem:

| Limit | Workers | Probabilistic Monkey (152.6M) | Big (516.9M) |
|---|---|---|---|
| Memory per isolate (JS heap + WebAssembly) | 128 MB | ~150 MB of int8 weights alone, ~770 MB as the engine runs them | ~520 MB int8, ~2.3 GB running |
| Worker bundle size | 64 MiB | weights are 150 MB | weights are ~520 MB |
| CPU time per request | 10 ms (Free) / 5 min (Paid) | a reply needs a few CPU-seconds | more |

A mixture-of-experts model touches a different expert for almost every token, so the weights cannot be streamed in a piece at a time. They all have to stay in memory, and they don't fit in 128 MB. Shrinking the small model to 4 bits (about 76 MB) and rewriting the engine in WebAssembly might squeeze under the limit. That would mean a second inference engine to maintain, a re-validation of the model's quality at 4 bits, almost no memory headroom for concurrent requests, and still no route for the Big model. It isn't practical.

[Cloudflare Containers](https://developers.cloudflare.com/containers/) exist for exactly this case: a Worker that needs more memory and CPU than an isolate can give. The container also uses the same tested server and engine as the local app, so nothing has to be re-implemented.

## What you need

* A Cloudflare account on the **Workers Paid plan** ($5/month). Containers are not available on the Free plan.
* [Node.js](https://nodejs.org/) 18 or newer.
* [Docker](https://docs.docker.com/get-docker/), running. Wrangler builds the container image locally and pushes it to Cloudflare's registry.

## Deploy

```bash
cd deploy/cloudflare
npm install
npx wrangler login          # opens a browser to authorise Wrangler
npm run deploy              # = npx wrangler deploy
```

Wrangler builds the image from the repository root, so it includes `app/` and `model/`. It pushes the image, deploys the Worker and prints its URL, for example `https://probabilistic-monkey.<your-subdomain>.workers.dev`. Open that URL to get the same terminal page as `./run.sh`.

The first deploy can take a few minutes while Cloudflare provisions the container. The first visit after a deploy, or after the container has gone to sleep, takes about 10–20 seconds while the container starts and loads the model. After that, replies stream as usual.

To deploy **Probabilistic Monkey Big** once its trained weights exist in `model_big/`:

```bash
npm run deploy:big          # = npx wrangler deploy -c wrangler.big.jsonc
```

This deploys a separate Worker called `probabilistic-monkey-big` on a 1 vCPU / 4 GiB container.

## Configuration

| File | What it does |
|---|---|
| `src/index.ts` | The Worker. It forwards every request to one container instance named `monkey`. |
| `Dockerfile` | The container image: `python:3.12-slim`, numpy and tokenizers, `app/`, and one model folder (the `MODEL_DIR` build argument). |
| `wrangler.jsonc` | The small model on a `standard-1` instance (½ vCPU, 4 GiB). |
| `wrangler.big.jsonc` | The Big model on a custom 1 vCPU / 4 GiB instance. |

Things you may want to change:

* **Instance size.** The small model also runs on the cheaper `basic` instance (¼ vCPU, 1 GiB), at about half the speed. Set `"instance_type": "basic"` in `wrangler.jsonc` if you want that.
* **Idle timeout.** `sleepAfter = "15m"` in `src/index.ts` decides how long the container stays awake after the last request. Shorter is cheaper; longer means fewer cold starts.
* **Concurrency.** One container generates one reply at a time, and other requests wait their turn. For more simultaneous users, raise `max_instances` and route users to different named instances in `src/index.ts`, for example `getContainer(env.MONKEY, "monkey-" + n)`.

## Tested here

* `npx tsc --noEmit` type-checks the Worker against `@cloudflare/workers-types`.
* `npx wrangler deploy --dry-run` accepts the configuration and starts the image build.
* The image was built with Docker and run under the same CPU and memory limits as Cloudflare's instance types. In a `standard-1`-sized container (½ vCPU, 4 GiB) the small model loads, serves the page, uses 772 MiB of RAM and streams about 22 tokens/s. In a `basic`-sized container (¼ vCPU, 1 GiB) it also runs, at about 11 tokens/s.

The actual `wrangler deploy` to a Cloudflare account was **not** run, because no Cloudflare account was available to Claude while building this. Treat the first deploy as the final check.

## Cost

Containers bill for every 10 ms they are awake. Memory and disk are billed on the provisioned size and CPU on actual use ([pricing](https://developers.cloudflare.com/containers/platform/pricing/)). At the time of writing, a `standard-1` container costs roughly $0.04 per hour awake beyond the included allowance. The Workers Paid plan includes 25 GiB-hours of memory a month, which is about 6 awake-hours of `standard-1` or 25 hours of `basic`. Because the container sleeps after 15 idle minutes, a lightly used demo costs little or nothing beyond the $5 plan.
