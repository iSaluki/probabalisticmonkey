# Deploying to Cloudflare

This folder deploys the chat page, Probabilistic Monkey and Probabilistic Gorilla to Cloudflare as a **Worker in front of a Cloudflare Container**. The Worker receives every request and forwards it to a container. The container runs the same NumPy server and page as `./run.sh`, including switching between the two models, and goes to sleep when nobody is using it.

## Why not a plain Worker?

A Worker on its own cannot run these models. Cloudflare's [Workers limits](https://developers.cloudflare.com/workers/platform/limits/) are the problem:

| Limit | Workers | Probabilistic Monkey (152.6M) | Probabilistic Gorilla (516.9M) |
|---|---|---|---|
| Memory per isolate (JS heap + WebAssembly) | 128 MB | ~150 MB of int8 weights alone, ~770 MB as the engine runs them | ~520 MB int8, ~2.3 GB running |
| Worker bundle size | 64 MiB | weights are 150 MB | weights are ~520 MB |
| CPU time per request | 10 ms (Free) / 5 min (Paid) | a reply needs a few CPU-seconds | more |

A mixture-of-experts model touches a different expert for almost every token, so the weights cannot be streamed in a piece at a time. They all have to stay in memory, and they don't fit in 128 MB. Shrinking Probabilistic Monkey to 4 bits (about 76 MB) and rewriting the engine in WebAssembly might squeeze under the limit. That would mean a second inference engine to maintain, a re-validation of the model's quality at 4 bits, almost no memory headroom for concurrent requests, and still no route for the Gorilla. It isn't practical.

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

Wrangler builds the image from the repository root and pushes it. The `.dockerignore` there keeps only `app/`, `model/` and, once it exists, `model_gorilla/`. Wrangler then deploys the Worker and prints its URL, for example `https://probabilistic-monkey.<your-subdomain>.workers.dev`. Open that URL to get the same terminal page as `./run.sh`. The page lists both models with their parameter counts, and you can click a name or type `/model gorilla` to switch.

Probabilistic Gorilla is included automatically if `model_gorilla/` exists when you deploy. Until it has been trained, the page shows it as "not installed". After it is trained, run `npm run deploy` again.

The first deploy can take a few minutes while Cloudflare provisions the container. The first visit after a deploy, or after the container has gone to sleep, takes about 10–20 seconds while the container starts and loads Probabilistic Monkey. The Gorilla is loaded the first time someone switches to it, which takes a few more seconds. After that, replies stream as usual.

## Configuration

| File | What it does |
|---|---|
| `src/index.ts` | The Worker. It forwards every request to one container instance named `monkey`. |
| `Dockerfile` | The container image: `python:3.12-slim`, numpy and tokenizers, `app/` and the model folders. |
| `wrangler.jsonc` | The container: a custom 1 vCPU / 4 GiB instance, enough to hold both models at once. |
| `../../.dockerignore` | Limits the image to the app and the trained models. |

Things you may want to change:

* **Instance size.** If you only want Probabilistic Monkey (delete or don't train `model_gorilla/`), the cheaper `basic` instance (¼ vCPU, 1 GiB) is enough: set `"instance_type": "basic"`. It runs at about a quarter of the speed.
* **Idle timeout.** `sleepAfter = "15m"` in `src/index.ts` decides how long the container stays awake after the last request. Shorter is cheaper; longer means fewer cold starts.
* **Concurrency.** One container generates one reply at a time, and other requests wait their turn. For more simultaneous users, raise `max_instances` and route users to different named instances in `src/index.ts`, for example `getContainer(env.MONKEY, "monkey-" + n)`.

## Tested here

* `npx tsc --noEmit` type-checks the Worker against `@cloudflare/workers-types`.
* `npx wrangler deploy --dry-run` accepts the configuration and starts the image build.
* The image was built with Docker from the repository root, and contained only `app/` and `model/`. It was run under the limits of the deployed instance (1 vCPU, 4 GiB). Probabilistic Monkey loaded and served the page, `/models` listed both models with their parameter counts, and replies streamed at about 50 tokens/s.
* Model switching was tested in a browser against the same server, both with a second model installed and with the Gorilla missing.
* Earlier, the image was also run at `standard-1` limits (½ vCPU, 4 GiB): 772 MiB of RAM, about 22 tokens/s. At `basic` limits (¼ vCPU, 1 GiB) it ran at about 11 tokens/s.

Not tested: the actual `wrangler deploy` to a Cloudflare account, because no Cloudflare account was available to Claude while building this. Also not tested: the Gorilla inside the container, because it is still training. Its ~2.3 GB is estimated from its parameter count (516.9M × 4 bytes once de-quantised, plus overhead). Treat the first deploy as the final check.

## Cost

Containers bill for every 10 ms they are awake. Memory and disk are billed on the provisioned size and CPU on actual use ([pricing](https://developers.cloudflare.com/containers/platform/pricing/)). At the time of writing, a 4 GiB container costs roughly $0.04 per hour awake beyond the included allowance. The Workers Paid plan includes 25 GiB-hours of memory a month, which is about 6 awake-hours at 4 GiB or 25 hours of `basic`. Because the container sleeps after 15 idle minutes, a lightly used demo costs little or nothing beyond the $5 plan.
