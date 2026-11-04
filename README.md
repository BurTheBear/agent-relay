# Agent Relay

I use Claude to help me code, and when I hit my usage limit or walk away, my projects just sit there. Agent Relay keeps them moving. When Claude has been quiet for 15 minutes, it starts one worker on the next task in my list. The worker can be a local model through Ollama, Codex, or headless Claude, in whatever order each project picks. As soon as I'm using Claude again, the worker stops and the local model is unloaded so I get my RAM back.

Workers never touch my main branch. Each project gets its own git worktree on a `relay/<project>` branch, and the worker isn't allowed to commit, merge or push. After it exits, the relay runs the project's checks (tests, linters, whatever I list), and only commits if every one passes. If they fail, the changes stay uncommitted and the next worker is told to fix them. I review the relay branch myself and merge what I want.

Other things it does:

- Stops a local model if it goes over a RAM limit I set.
- Waits out usage limits and backs off after failures, with a cap on runs per day.
- Has a task queue per project, so when one task is done the next one starts.
- Only counts Claude activity in my own sessions, not the relay's worktrees, so it doesn't pause itself.

It's one Python file with no dependencies outside the standard library.

## The web app

`python3 relay.py serve` runs the controller and a small web app at http://127.0.0.1:8777. It lists every project and lets me add projects, pause, run or stop a worker, edit the next task, and see the commits on the relay branch. It only listens on localhost, and every request needs a random token that's written into the page when it loads.

## Using it

```bash
python3 relay.py add myproject ~/code/myproject   # needs at least one commit
python3 relay.py serve                            # controller + web app
python3 relay.py status
python3 relay.py pause                            # also stops a running worker
python3 relay.py resume
```

Projects start disabled. Turn one on and give it a task in the web app. Paths, checks and extra environment variables are set by hand in `projects/<name>/project.json`.

On my Mac it runs from a launchd agent under `caffeinate -i`, so it keeps going while the screen is off.

`setup_model.py` picks a local model. For each candidate it downloads it, makes a version with a bigger context window, measures how much memory it really uses, and gives it a small coding task through Codex. The first one that passes becomes the default.

## Tests

```bash
python3 -m unittest discover -s tests
```

MIT license.
