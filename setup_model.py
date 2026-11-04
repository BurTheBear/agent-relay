#!/usr/bin/env python3
"""One-time, unattended local-model setup for Agent Relay.

For each candidate (best first): download it, make a ``-relay`` variant with a large context window, load it
and measure its real memory with ``ollama ps``, shrink the context if it is over the RAM limit, then give it a
small real coding task through the same Codex command the relay uses. The first candidate that passes becomes
the relay's default model. Everything is written to model-setup.log. It never deletes a model.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "model-setup.log"
OLLAMA = "/opt/homebrew/bin/ollama"
CODEX = str(Path.home() / ".local/bin/codex")
CANDIDATES = sys.argv[1:] or ["qwen3.8:27b", "flint:40b"]
CONTEXTS = [65536, 49152, 32768]


def log(msg: str) -> None:
    with LOG.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def run(args, timeout, **kw):
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout, **kw)


def loaded_gb(name: str) -> float:
    with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=10) as r:
        for m in json.load(r).get("models", []):
            if m["name"].split(":")[0] == name.split(":")[0]:
                return m.get("size", 0) / 1e9
    return 0.0


def settings_update(**fields) -> None:
    path = ROOT / "settings.json"
    s = json.loads(path.read_text()) if path.exists() else {}
    s.update(fields)
    tmp = path.with_suffix(".setup.tmp")
    tmp.write_text(json.dumps(s, indent=2) + "\n")
    tmp.replace(path)


def limit_gb() -> float:
    path = ROOT / "settings.json"
    return float(json.loads(path.read_text()).get("ram_limit_gb", 42.67)) if path.exists() else 42.67


def prepare(base: str) -> str | None:
    for attempt in range(6):  # downloads resume after network drops
        p = run([OLLAMA, "pull", base], timeout=6 * 3600)
        if p.returncode == 0:
            break
        log(f"pull {base} failed (attempt {attempt + 1}): {p.stderr.strip()[-300:]}")
        time.sleep(60)
    else:
        return None
    log(f"pulled {base}")
    variant = base.split(":")[0] + "-relay"
    for ctx in CONTEXTS:
        mf = Path(tempfile.mkdtemp()) / "Modelfile"
        mf.write_text(f"FROM {base}\nPARAMETER num_ctx {ctx}\n")
        c = run([OLLAMA, "create", variant, "-f", str(mf)], timeout=600)
        if c.returncode:
            log(f"create {variant} failed: {c.stderr.strip()[-300:]}")
            return None
        t0 = time.time()
        r = run([OLLAMA, "run", variant, "--keepalive", "5m", "Reply with the single word ready."], timeout=900)
        gb = loaded_gb(variant)
        run([OLLAMA, "stop", variant], timeout=60)
        log(f"{variant} num_ctx={ctx}: loaded {gb:.2f} GB, first reply in {time.time() - t0:.0f} s: "
            f"{r.stdout.strip()[-80:]!r}")
        if 0 < gb <= limit_gb():
            return variant
    log(f"{variant}: over the RAM limit at every context size")
    return None


def smoke_test(variant: str) -> bool:
    """A real tool-using task through Codex: create a file and run a test, in a scratch git repo."""
    work = Path(tempfile.mkdtemp(prefix="relay-smoke-"))
    run(["git", "init", "-q"], timeout=30, cwd=work)
    (work / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (work / "test_calc.py").write_text("from calc import add\n\nassert add(2, 3) == 5\nprint('ok')\n")
    task = ("calc.py has a bug: test_calc.py fails. Fix calc.py (do not edit the test), then run "
            "`python3 test_calc.py` to confirm it passes. Reply with the test result.")
    t0 = time.time()
    p = run([CODEX, "exec", "--ignore-user-config", "--approve-for-me", "--oss", "--local-provider", "ollama",
             "--model", variant, "--sandbox", "workspace-write", "--cd", str(work), "--skip-git-repo-check", task],
            timeout=1800)
    run([OLLAMA, "stop", variant], timeout=60)
    fixed = "a + b" in (work / "calc.py").read_text()
    untouched = "assert add(2, 3) == 5\n" in (work / "test_calc.py").read_text()
    check = run([sys.executable, "test_calc.py"], timeout=120, cwd=work) if fixed else None
    ok = bool(fixed and untouched and check is not None and check.returncode == 0)
    log(f"smoke {variant}: exit {p.returncode}, {time.time() - t0:.0f} s, fixed={fixed}, test untouched={untouched},"
        f" passes={ok}. Tail: {(p.stdout + p.stderr).strip()[-400:]!r}")
    return ok


def main() -> None:
    log(f"setup start: candidates {CANDIDATES}")
    for base in CANDIDATES:
        variant = prepare(base)
        if variant and smoke_test(variant):
            settings_update(local_model=variant)
            log(f"DEFAULT MODEL SET: {variant}")
            return
    log("no candidate passed; relay local model left unset (relay waits)")


if __name__ == "__main__":
    main()
