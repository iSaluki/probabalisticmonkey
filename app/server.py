"""Tiny local web server for chatting with Probabilistic Monkey.

    python app/server.py [--port 8642] [--no-browser]

Serves a single terminal-style page at http://127.0.0.1:8642 . Only the
Python standard library, numpy and tokenizers are needed.
"""
import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Monkey  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = open(os.path.join(HERE, "index.html"), "rb").read()
MODEL = None
LOCK = threading.Lock()  # one generation at a time keeps the CPU from thrashing


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(PAGE)))
        self.end_headers()
        self.wfile.write(PAGE)

    def do_POST(self):
        if self.path != "/chat":
            self.send_error(404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            messages = [{"role": m["role"], "content": str(m["content"])} for m in body["messages"]
                        if m.get("role") in ("user", "assistant")]
            if not messages or messages[-1]["role"] != "user":
                raise ValueError("last message must be from the user")
        except Exception as e:  # malformed request
            self.send_error(400, str(e))
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with LOCK:
            try:
                for piece in MODEL.generate(messages):
                    self.wfile.write(piece.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    print("Loading Probabilistic Monkey...", flush=True)
    MODEL = Monkey()
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}/"
    print(f"Probabilistic Monkey is running at {url}  (press Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
