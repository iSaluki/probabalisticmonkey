"""Tiny local web server for chatting with Probabilistic Monkey and Probabilistic Gorilla.

    python app/server.py [--port 8642] [--host 127.0.0.1] [--gorilla] [--no-browser]

Serves a single terminal-style page at http://127.0.0.1:8642 . Every model
whose folder exists can be chosen from the page; each one is loaded the first
time it is used. Only the Python standard library, numpy and tokenizers are needed.
"""
import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Monkey, param_counts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PAGE = open(os.path.join(HERE, "index.html"), "rb").read()

# id -> display name, folder, and parameter counts used when the folder is missing
MODELS = {
    "monkey": {"name": "Probabilistic Monkey", "dir": "model", "params": (152_619_520, 42_519_040)},
    "gorilla": {"name": "Probabilistic Gorilla", "dir": "model_gorilla", "params": (516_890_240, 74_522_240)},
}
LOADED = {}
LOAD_LOCK = threading.Lock()
GEN_LOCK = threading.Lock()  # one generation at a time keeps the CPU from thrashing
DEFAULT = "monkey"


def model_dir(mid):
    return os.path.join(REPO, MODELS[mid]["dir"])


def available(mid):
    return os.path.exists(os.path.join(model_dir(mid), "config.json"))


def info(mid):
    m = MODELS[mid]
    total, active = m["params"]
    if available(mid):
        with open(os.path.join(model_dir(mid), "config.json")) as f:
            total, active = param_counts(json.load(f))
    return {"id": mid, "name": m["name"], "available": available(mid), "loaded": mid in LOADED,
            "params_total": total, "params_active": active}


def get_model(mid):
    """Load a model the first time it is asked for."""
    with LOAD_LOCK:
        if mid not in LOADED:
            print(f"Loading {MODELS[mid]['name']}...", flush=True)
            LOADED[mid] = Monkey(model_dir(mid))
        return LOADED[mid]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    def pick_model(self, body):
        mid = body.get("model", DEFAULT)
        if mid not in MODELS:
            raise ValueError(f"unknown model '{mid}'")
        if not available(mid):
            raise ValueError(f"{MODELS[mid]['name']} is not installed (no {MODELS[mid]['dir']}/ folder)")
        return mid

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/models":
            self.send_json(200, {"default": DEFAULT, "models": [info(m) for m in MODELS]})
            return
        if path not in ("/", "/index.html"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(PAGE)))
        self.end_headers()
        self.wfile.write(PAGE)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/load":
            try:
                mid = self.pick_model(self.read_json())
                get_model(mid)
                self.send_json(200, info(mid))
            except ValueError as e:
                self.send_json(400, {"error": str(e)})
            return
        if path != "/chat":
            self.send_error(404)
            return
        try:
            body = self.read_json()
            mid = self.pick_model(body)
            messages = [{"role": m["role"], "content": str(m["content"])} for m in body["messages"]
                        if m.get("role") in ("user", "assistant")]
            if not messages or messages[-1]["role"] != "user":
                raise ValueError("last message must be from the user")
        except Exception as e:  # malformed request
            self.send_error(400, str(e))
            return
        model = get_model(mid)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with GEN_LOCK:
            try:
                for piece in model.generate(messages):
                    self.wfile.write(piece.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass


def main():
    global DEFAULT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--gorilla", action="store_true", help="start with Probabilistic Gorilla selected")
    a = ap.parse_args()
    installed = [m for m in MODELS if available(m)]
    if not installed:
        print("No trained model was found (model/ or model_gorilla/ with config.json and weights-*.npz).\n"
              "Download a version of the repository that includes one, or train one with the pipeline\n"
              "in training/ (see README.md).", file=sys.stderr)
        sys.exit(1)
    DEFAULT = "gorilla" if a.gorilla else "monkey"
    if DEFAULT not in installed:
        print(f"{MODELS[DEFAULT]['name']} is not installed; starting with {MODELS[installed[0]]['name']}.",
              file=sys.stderr)
        DEFAULT = installed[0]
    get_model(DEFAULT)
    for m in MODELS:
        i = info(m)
        print(f"  {i['name']}: {i['params_total'] / 1e6:.1f}M parameters ({i['params_active'] / 1e6:.1f}M active)"
              f"{'' if i['available'] else '  [not installed]'}", flush=True)
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}/"
    print(f"Running at {url}  (press Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
