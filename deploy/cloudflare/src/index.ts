// Cloudflare Worker that serves Probabilistic Monkey and Probabilistic Gorilla from a Cloudflare Container.
//
// A Worker isolate cannot hold the models (128 MB memory limit), so the
// Worker forwards every request to a container running the same NumPy server
// as ./run.sh, including model switching. The container sleeps when idle, so it only
// costs money while in use.
import { Container, getContainer } from "@cloudflare/containers";

export class MonkeyContainer extends Container {
  // app/server.py listens here once the model is loaded, so requests wait for it
  defaultPort = 8080;
  // shut down after 15 idle minutes; the next visitor starts it again (cold start ~10-20 s)
  sleepAfter = "15m";
}

interface Env {
  MONKEY: DurableObjectNamespace<MonkeyContainer>;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // One shared instance: the server generates one reply at a time, which is
    // plenty for a demo. Use a different name per user to give each their own.
    return getContainer(env.MONKEY, "monkey").fetch(request);
  },
};
