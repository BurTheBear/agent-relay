"""Agent Relay tests: scheduling, preemption, checks, isolation and the web app's access rules.

No model is called. Workers are replaced by small Python scripts; Ollama and Claude transcripts are faked.
Run: python3 -m unittest discover -s tests
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

TMP = tempfile.TemporaryDirectory()
os.environ["AGENT_RELAY_HOME"] = str(Path(TMP.name) / "home")
os.environ["AGENT_RELAY_TRANSCRIPTS"] = str(Path(TMP.name) / "transcripts")
spec = importlib.util.spec_from_file_location("relay", Path(__file__).parents[1] / "relay.py")
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)
relay.POLL_SECONDS = 0.1

WRITE_AND_FINISH = (
    "import sys, pathlib; sys.stdin.read(); pathlib.Path('feature.txt').write_text('done');"
    "pathlib.Path(sys.argv[1]).write_text('Added feature.txt. Next: tests.')")
SLEEP = "import sys, time; sys.stdin.read(); time.sleep(60)"


def make_repo(root: Path, name: str) -> Path:
    repo = root / name
    repo.mkdir(parents=True)
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@example.com"],
                 ["config", "user.name", "T"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    (repo / "README.md").write_text("hi\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    return repo


class RelayTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(dir=TMP.name))
        (relay.PROJECTS / ".removed").mkdir(parents=True, exist_ok=True)
        for p in relay.project_ids():
            (relay.PROJECTS / p).rename(relay.PROJECTS / ".removed" / f"{p}-{time.time()}")
        relay.save(relay.SETTINGS, {"enabled": True, "local_model": "fake:1b", "quiet_seconds": 60})
        self.repo = make_repo(self.root, "app")
        self.pid = relay.add_project("App", str(self.repo), checks=[[sys.executable, "-c", "pass"]])
        relay.update(relay.project_dir(self.pid) / "project.json", lambda c: c.update(enabled=True))
        self.patches = [patch.object(relay, "ollama_models", return_value=["fake:1b"]),
                        patch.object(relay, "ollama_loaded", return_value=[]),
                        patch.object(relay, "ollama_stop")]
        self.stop = [p.start() for p in self.patches][2]
        self.claude_last = patch.object(relay, "claude_last_active", return_value=0.0)
        self.claude = self.claude_last.start()
        self.script = WRITE_AND_FINISH
        self.cmd = patch.object(relay, "agent_command",
                                side_effect=lambda prov, cfg, s, last: [sys.executable, "-c", self.script, str(last)])
        self.cmd.start()

    def tearDown(self):
        patch.stopall()

    def state(self):
        return relay.read(relay.state_path(self.pid))

    def test_queue_feeds_next_task_in_order(self):
        relay.update(relay.project_dir(self.pid) / "project.json",
                     lambda c: c.update(next_task="", queue=["first", "second"]))
        relay.tick()
        cfg = relay.project(self.pid)
        self.assertEqual((cfg["next_task"], cfg["queue"]), ("second", []))

    def test_waits_while_claude_is_active(self):
        self.claude.return_value = time.time() - 10
        relay.tick()
        self.assertEqual(self.state()["phase"], "waiting_for_quiet")
        self.assertFalse((self.repo.parent / "app-relay-work").exists())

    def test_global_pause_and_project_pause(self):
        relay.update(relay.SETTINGS, lambda c: c.update(enabled=False))
        relay.tick()
        self.assertEqual(self.state()["phase"], "paused")
        relay.update(relay.SETTINGS, lambda c: c.update(enabled=True))
        relay.update(relay.project_dir(self.pid) / "project.json", lambda c: c.update(enabled=False))
        relay.tick()
        self.assertEqual(self.state()["phase"], "paused")

    def test_quiet_runs_worker_commits_to_relay_branch_only(self):
        relay.update(relay.project_dir(self.pid) / "project.json", lambda c: c.update(next_task="add feature"))
        relay.tick()
        st = self.state()
        self.assertEqual(st["phase"], "completed", st.get("reason"))
        work = Path(relay.project(self.pid)["worktree"])
        self.assertTrue((work / "feature.txt").exists())
        self.assertFalse((self.repo / "feature.txt").exists())  # main checkout untouched
        log = subprocess.run(["git", "-C", str(self.repo), "log", "--oneline", "main..relay/app"],
                             capture_output=True, text=True).stdout
        self.assertIn("relay: local milestone", log)
        self.assertEqual(relay.project(self.pid)["next_task"], "")  # a finished task is cleared
        self.stop.assert_called_with("fake:1b")  # model unloaded after the run

    def test_claude_returning_preempts_and_unloads_model(self):
        self.script = SLEEP
        start = time.time()
        calls = {"n": 0}

        def active(cfg, s, window):
            calls["n"] += 1
            return 0.0 if calls["n"] == 1 else time.time()  # quiet at dispatch, active once running
        self.claude.side_effect = active
        relay.tick()
        self.assertEqual(self.state()["phase"], "preempted")
        self.assertLess(time.time() - start, 20)
        self.stop.assert_called_with("fake:1b")

    def test_forced_run_ignores_claude_activity_until_stopped(self):
        self.claude.return_value = time.time()
        (relay.project_dir(self.pid) / "run.flag").write_text("x")
        self.script = WRITE_AND_FINISH
        relay.tick()
        self.assertEqual(self.state()["phase"], "completed")

    def test_stop_flag_stops_worker(self):
        self.script = SLEEP

        def active(cfg, s, window):
            (relay.project_dir(self.pid) / "stop.flag").write_text("x")
            return 0.0
        self.claude.side_effect = active
        relay.tick()
        self.assertEqual(self.state()["phase"], "stopped")

    def test_ram_limit_stops_local_model(self):
        self.script = SLEEP
        self.patches[1].stop()
        with patch.object(relay, "ollama_loaded", return_value=[{"name": "fake:1b", "gb": 50.0, "context": 1}]):
            relay.tick()
        self.assertEqual(self.state()["phase"], "ram_limit")

    def test_failed_checks_leave_work_uncommitted(self):
        relay.update(relay.project_dir(self.pid) / "project.json",
                     lambda c: c.update(checks=[[sys.executable, "-c", "raise SystemExit(1)"]]))
        relay.tick()
        self.assertEqual(self.state()["phase"], "checks_failed")
        log = subprocess.run(["git", "-C", str(self.repo), "log", "--oneline", "main..relay/app"],
                             capture_output=True, text=True).stdout
        self.assertEqual(log.strip(), "")

    def test_missing_model_means_no_provider(self):
        self.patches[0].stop()
        with patch.object(relay, "ollama_models", return_value=[]):
            relay.tick()
        self.assertEqual(self.state()["phase"], "no_provider")

    def test_one_worker_at_a_time_round_robin(self):
        other = make_repo(self.root, "other")
        pid2 = relay.add_project("Other", str(other))
        relay.update(relay.project_dir(pid2) / "project.json", lambda c: c.update(enabled=True))
        relay.tick()
        relay.tick()
        self.assertEqual(self.state()["phase"], "completed")
        self.assertEqual(relay.read(relay.state_path(pid2))["phase"], "completed")

    def test_main_moved_with_uncommitted_relay_work_needs_review(self):
        relay.update(relay.project_dir(self.pid) / "project.json",
                     lambda c: c.update(checks=[[sys.executable, "-c", "raise SystemExit(1)"]]))
        relay.tick()  # leaves uncommitted work
        (self.repo / "x.txt").write_text("x")
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "move"], check=True)
        relay.update(relay.project_dir(self.pid) / "project.json", lambda c: c.update(failure_cooldown_seconds=0))
        st = self.state()
        st["cooldowns"] = {}
        relay.save(relay.state_path(self.pid), st)
        relay.tick()
        self.assertEqual(self.state()["phase"], "needs_attention")


class TranscriptTests(unittest.TestCase):
    def test_scope_and_relay_worktree_exclusion(self):
        t = relay.TRANSCRIPTS
        (t / "-Users-x").mkdir(parents=True, exist_ok=True)
        f = t / "-Users-x" / "s.jsonl"
        f.write_text('{"text": "working on Some Other thing"}\n')
        s = {**relay.DEFAULT_SETTINGS, "claude_activity_scope": "any"}
        self.assertGreater(relay.claude_last_active({"keywords": ["app"]}, s, 60), 0)
        s["claude_activity_scope"] = "project"
        self.assertEqual(relay.claude_last_active({"keywords": ["kodiak"]}, s, 60), 0)
        f.write_text('{"text": "Kodiak search tweak"}\n')
        self.assertGreater(relay.claude_last_active({"keywords": ["kodiak"]}, s, 60), 0)
        old = time.time() - 3600
        os.utime(f, (old, old))
        self.assertEqual(relay.claude_last_active({"keywords": ["kodiak"]}, s, 60), 0)


class WebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = relay.serve_ui(0)
        cls.port = cls.server.server_address[1]
        cls.tok = relay.token()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def req(self, path, body=None, token=True, host=None):
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                   data=None if body is None else json.dumps(body).encode(),
                                   method="GET" if body is None else "POST")
        if token:
            r.add_header("X-Relay-Token", self.tok)
        if host:
            r.add_header("Host", host)
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def test_page_carries_token_and_api_requires_it(self):
        code, html = self.req("/", token=False)
        self.assertEqual(code, 200)
        self.assertIn(self.tok, html)
        self.assertEqual(self.req("/api/status", token=False)[0], 403)
        self.assertEqual(self.req("/api/settings", {"enabled": False}, token=False)[0], 403)
        self.assertEqual(self.req("/api/status")[0], 200)

    def test_foreign_host_rejected(self):
        self.assertEqual(self.req("/", token=False, host="evil.example:80")[0], 403)
        self.assertEqual(self.req("/api/status", host="evil.example")[0], 403)

    def test_add_edit_run_remove_project(self):
        repo = make_repo(Path(tempfile.mkdtemp(dir=TMP.name)), "webapp")
        code, body = self.req("/api/projects", {"name": "Web App", "repo": str(repo)})
        self.assertEqual(code, 200, body)
        pid = json.loads(body)["id"]
        self.assertEqual(relay.project(pid)["enabled"], False)  # new projects start paused
        self.req(f"/api/projects/{pid}", {"enabled": True, "provider_order": ["codex", "bogus", "local", "codex"],
                                          "repo": "/etc", "next_task": "do x"})
        cfg = relay.project(pid)
        self.assertEqual(cfg["provider_order"], ["codex", "local"])
        self.assertEqual(cfg["repo"], str(repo.resolve()))  # paths are not editable from the web app
        self.assertEqual(cfg["next_task"], "do x")
        self.req(f"/api/projects/{pid}/run", {})
        self.assertTrue((relay.project_dir(pid) / "run.flag").exists())
        self.req(f"/api/projects/{pid}/remove", {})
        self.assertNotIn(pid, relay.project_ids())
        self.assertTrue(repo.exists())

    def test_rejects_non_repo_and_bad_handoff_names(self):
        self.assertEqual(self.req("/api/projects", {"name": "x", "repo": TMP.name})[0], 400)
        self.assertIn(self.req("/api/projects/nope/handoff?name=../../etc/passwd")[0], (400, 404))


if __name__ == "__main__":
    unittest.main()
