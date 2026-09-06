# AGENTS.md

## Project

- MotionDetector is a **Flet** (Python) desktop/mobile/web app, scaffolded from the Flet template. The code is still the counter demo — no motion-detection logic exists yet (`description` is empty, only dependency is `flet`).
- Package manager is **uv** (`pyproject.toml`, uv.lock). Use `uv run ...`, `uv add`, etc. — never pip directly.

## Run

```bash
uv run flet run          # desktop app
uv run flet run --web    # web app
```

- App entrypoint is `src/main.py` (set via `[tool.flet.app] path = "src"`; there is no root-level `main.py`). Change `path` if you move it.

## Test

- Tests live in `tests/` (`[tool.pytest.ini_options] testpaths`), async mode is auto.
- Requires the dev extra `flet[test]`; run with `flet test` or `uv run pytest`.
- Do not write your own `flet_app` fixture — the flet pytest plugin injects it. Follow `tests/test_main.py` (use `tester.pump_and_settle()`, `find_by_key`, `find_by_text`).

## Build

- `flet build apk -v` for Android (also `ipa`, `macos`, `linux`, `windows`, `web`). Uses `[tool.flet]` org/product metadata for signing/bundle id.

## Git / repo layout

- This folder has **no `.git`** — the git repo root is the Windows home dir (`C:\Users\david`), which must never be staged/committed. Do not `git init`, `git add`, or commit here; just edit files.

## Tooling

- No CI, lint, or typecheck setup exists; don't add tooling that isn't already present in sibling Python projects.
- `.gitignore` covers Python caches, `.venv`, `.flet/`, `build/`, `storage/`.