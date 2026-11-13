#!/usr/bin/env python3
"""strata-prefix-cache - keep agent clients' fixed prompt starts (tool list + the stable part of the system prompt)
as Strata session files on disk, and load them into a running server, so a new session does not read ~12K tokens
of system prompt again (about 40 s on one P100).

How it works, with Strata's own features only (docs/DETAILS.md: "Session files", "Multiple conversations",
"strata_prefix"):
  learn    read the requests the server saw (its API monitor, `"api_monitor": true`) and keep one template per
           distinct prompt start in DIR/<key>.request.json. The key is a hash of the tools and the system prompt up
           to its first per-session part (Hermes: the memory block; other clients: the whole system message).
  capture  the same, for Hermes, without the model: runs `hermes chat --oneshot` against a stub endpoint here that
           records the request and answers "ok". (A CLI request; the TUI's prompt differs - use `learn` for that.)
  build    for each template without a session file: send the prompt start once with `strata_prefix` (the engine
           pins a checkpoint exactly where the per-session part begins), then save the session to <key>.bin.
  load     restore every session file (each parks in the conversation cache; the newest is pinned so it is
           never evicted). New conversations that start the same way borrow the prefix from there.
  status   the templates and session files.
  rm KEY   delete one template and its session file.

The server needs, in its config: "slot_save_path": DIR, "api_monitor": true (for learn), and the engine args
--conversation-cache-mib N (room for the loaded prefixes plus real conversations; ~0.45 GB per 12K-token prefix).
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = os.environ.get("STRATA_URL", "http://127.0.0.1:8793")
DIR = Path(os.environ.get("STRATA_PREFIX_DIR", "/storage/fastSATA/strata/sessions"))
MODEL = os.environ.get("STRATA_MODEL", "qwen3.8-flash-next-iq3s-strata")
MIN_TOKENS_GUESS = 2048 * 3          # characters; shorter prompt starts are not worth a file
HERMES_MEMORY_RULE = "\n" + "\u2550" * 10   # Hermes: the "══════" line that opens the MEMORY block
# Hermes system-prompt sections that differ per session or change often, in any order: the project's AGENTS.md/
# CLAUDE.md, the git snapshot of the working directory, the skills list (new or edited skills), memory, and the
# runtime environment (date, cwd). The prefix ends at the first.
HERMES_SESSION_MARKERS = ("\n# Project Context\n", "\nWorkspace (snapshot", "\n## Skills\n", HERMES_MEMORY_RULE,
                          "\n# Hermes runtime environment")


def http_json(method, path, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def cut_point(system: str) -> tuple[int, str]:
    """Where the per-session part of a system prompt begins: (chars, rule)."""
    if system.startswith("You are Hermes Agent"):
        hits = [(system.find(m), m) for m in HERMES_SESSION_MARKERS]
        hits = [(i, m) for i, m in hits if i > 0]
        if hits:
            i, m = min(hits)
            return i, f"hermes: before {m.strip().split(chr(10))[0][:30]!r}"
    return len(system), "whole system message"


def template_of(body: dict) -> dict | None:
    msgs = body.get("messages") or []
    if not msgs or msgs[0].get("role") not in ("system", "developer") or not isinstance(msgs[0].get("content"), str):
        return None
    system = msgs[0]["content"]
    n, rule = cut_point(system)
    tools = body.get("tools") or []
    if n + len(json.dumps(tools)) < MIN_TOKENS_GUESS:
        return None
    key = hashlib.sha256(json.dumps([tools, system[:n], body.get("reasoning_effort")], sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()[:16]
    client = "hermes" if system.startswith("You are Hermes Agent") else "other"
    return {"key": key, "client": client, "rule": rule, "cut": n, "system_chars": len(system), "tools": len(tools),
            "body": {k: body[k] for k in ("messages", "tools", "reasoning_effort", "chat_template_kwargs") if k in body}}


def save_template(t: dict, source: str) -> bool:
    DIR.mkdir(parents=True, exist_ok=True)
    p = DIR / f"{t['key']}.request.json"
    if p.exists():
        return False
    t = dict(t, source=source, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    p.write_text(json.dumps(t, ensure_ascii=False))
    os.chmod(p, 0o600)
    return True


def cmd_learn(a):
    st, listing = http_json("GET", "/api/requests")
    if st != 200:
        sys.exit(f"the server's API monitor is off or unreachable ({st}): set \"api_monitor\": true in the config")
    new = 0
    for r in listing.get("requests", []):
        if not str(r.get("path", "")).endswith("/chat/completions"):
            continue
        st, rec = http_json("GET", f"/api/requests?id={r['id']}")
        if st != 200 or rec.get("input_truncated"):
            continue
        try:
            body = json.loads(rec["input"])
        except (KeyError, ValueError):
            continue
        t = template_of(body)
        if t and save_template(t, "learn"):
            new += 1
            print(f"new template {t['key']} ({t['client']}, {t['tools']} tools, cut at {t['cut']} of "
                  f"{t['system_chars']} chars: {t['rule']})")
    print(f"{new} new template(s)")


class _Stub(http.server.BaseHTTPRequestHandler):
    captured: list = []

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": MODEL, "object": "model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        _Stub.captured.append(body)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for d in ({"role": "assistant", "content": "ok"}, {}):
                chunk = {"id": "stub", "object": "chat.completion.chunk", "model": MODEL,
                         "choices": [{"index": 0, "delta": d, "finish_reason": None if d else "stop"}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b'data: {"id":"stub","object":"chat.completion.chunk","choices":[],'
                             b'"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\ndata: [DONE]\n\n')
        else:
            self._send({"id": "stub", "object": "chat.completion", "model": MODEL,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


def cmd_capture(a):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    cmd = ["hermes", "chat", "--provider", "custom", "--base-url", base, "-m", MODEL, "--oneshot", "-Q", "-q", "ping",
           *a.hermes_args]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300, cwd=a.cwd or os.getcwd())
    srv.shutdown()
    new = 0
    for body in _Stub.captured:
        t = template_of(body)
        if t and save_template(t, "capture"):
            new += 1
            print(f"new template {t['key']} ({t['tools']} tools, cut at {t['cut']} of {t['system_chars']} chars)")
    print(f"captured {len(_Stub.captured)} request(s), {new} new template(s)")


def warm_body(t: dict) -> dict:
    """The template's prompt start, a stand-in for the per-session part, and the pin at the boundary."""
    b = json.loads(json.dumps(t["body"]))
    system = b["messages"][0]["content"]
    n = t["cut"]
    # keep the boundary's own characters so the tokens up to the cut are the real prompt's
    tail = system[n:n + len(HERMES_MEMORY_RULE)] if t["cut"] < len(system) else ""
    b["messages"] = [{"role": b["messages"][0]["role"], "content": system[:n] + tail + "\n(prefix cache)"},
                     {"role": "user", "content": "."}]
    b.update(model=MODEL, stream=False, max_tokens=1, temperature=0,
             strata_prefix={"message": 0, "chars": n})
    return b


def pin(t: dict) -> str:
    st, r = http_json("POST", "/v1/chat/completions", warm_body(t))
    return f"{st}" if st == 200 else f"{st} {r.get('error', {}).get('message', '')}"


def templates():
    return sorted((json.loads(p.read_text()) for p in DIR.glob("*.request.json")), key=lambda t: t["created"])


def cmd_build(a):
    for t in templates():
        f = DIR / f"{t['key']}.bin"
        if (a.key and t["key"] != a.key) or (f.exists() and not a.force):
            continue
        t0 = time.time()
        res = pin(t)
        if res != "200":
            print(f"{t['key']}: warm-up request failed: {res}")
            continue
        st, r = http_json("POST", "/slots/0?action=save", {"filename": f.name})
        if st != 200:
            print(f"{t['key']}: save failed ({st}): {r.get('error', {}).get('message', r)}")
            continue
        print(f"{t['key']}: {r['n_saved']} tokens, {r['n_written'] / 1e6:.0f} MB, {time.time() - t0:.0f} s")
    # each warm-up took the engine's one pin: give it back to the primary template
    primary = (DIR / "primary").read_text().strip() if (DIR / "primary").exists() else None
    p = next((t for t in templates() if t["key"] == primary), None)
    if p and (DIR / f"{primary}.bin").exists():
        print(f"pinned {primary} (primary): {pin(p)}")


def cmd_load(a):
    ts = [t for t in templates() if (DIR / f"{t['key']}.bin").exists()]
    if not ts:
        print("no session files")
        return
    primary = (DIR / "primary").read_text().strip() if (DIR / "primary").exists() else None
    ts.sort(key=lambda t: t["key"] == primary)          # the primary last: it stays live and gets the one pin
    for t in ts:
        st, r = http_json("POST", "/slots/0?action=restore", {"filename": f"{t['key']}.bin"})
        if st != 200:
            print(f"{t['key']}: restore failed ({st}): {r.get('error', {}).get('message', r)}")
            continue
        print(f"{t['key']}: restored {r['n_restored']} tokens in {r['timings']['restore_ms'] / 1000:.1f} s")
    print(f"pinned {ts[-1]['key']}: {pin(ts[-1])}")


def cmd_primary(a):
    if not (DIR / f"{a.key}.request.json").exists():
        sys.exit(f"no template {a.key}")
    (DIR / "primary").write_text(a.key + "\n")
    print(f"primary: {a.key} (loaded last and pinned on the next load)")


def cmd_status(a):
    primary = (DIR / "primary").read_text().strip() if (DIR / "primary").exists() else None
    for t in templates():
        f = DIR / f"{t['key']}.bin"
        size = f"{f.stat().st_size / 1e6:.0f} MB" if f.exists() else "not built"
        print(f"{t['key']}{'*' if t['key'] == primary else ' '} {t['client']:6}  {t['tools']:2} tools  "
              f"cut {t['cut']:6}/{t['system_chars']:6}  {t['source']:11}  {t['created']}  {size}")
    print("(* = primary: pinned, never evicted; the others can be evicted by real conversations until the next load)")


def cmd_rm(a):
    for suffix in (".request.json", ".bin"):
        p = DIR / f"{a.key}{suffix}"
        if p.exists():
            p.unlink()
            print(f"removed {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("learn")
    c = sub.add_parser("capture")
    c.add_argument("--cwd", help="directory to run hermes in")
    c.add_argument("hermes_args", nargs="*", help="extra `hermes chat` arguments (after --)")
    b = sub.add_parser("build")
    b.add_argument("key", nargs="?")
    b.add_argument("--force", action="store_true", help="rebuild existing session files")
    sub.add_parser("load")
    sub.add_parser("status")
    r = sub.add_parser("rm")
    r.add_argument("key")
    p = sub.add_parser("primary", help="the template to pin (one pin per engine)")
    p.add_argument("key")
    a = ap.parse_args()
    {"learn": cmd_learn, "capture": cmd_capture, "build": cmd_build, "load": cmd_load, "status": cmd_status,
     "rm": cmd_rm, "primary": cmd_primary}[a.cmd](a)


if __name__ == "__main__":
    main()
