# AI Exercise Mentor (Beta)

Thanks for testing! This is a **Practice-only beta build**: view and start a
lab, chat with an AI mentor while you work, finish and see your results,
and import an exercise file if someone sends you one. Everything else in
the full product (instructor tools, exercise authoring, class analytics,
settings, mentor review) is intentionally not part of this build.

## Data & Privacy

**Everything runs on your own machine. Nothing is sent anywhere unless you
manually share a file yourself.**

- The app's only network call is to `OLLAMA_BASE_URL`, which defaults to
  `localhost` -- your own local Ollama install. There is no telemetry,
  analytics, or crash reporting anywhere in this build.
- No cloud AI providers are used -- Ollama, running locally, is the only
  inference endpoint.
- All data (sessions, evaluations) lives in a local SQLite file
  (`mentor.db`) on your machine. Exercise definitions are local YAML files.
- The **only** way data leaves this app is if you deliberately click
  "Download Result to Share" on the results page and then send that file
  to someone yourself. The app never does this automatically.

## What You'll Need

| Component | Why | Install |
|---|---|---|
| Python 3.10+ | Runs the FastAPI app | [python.org](https://www.python.org/downloads/); macOS `brew install python`, Windows `winget install Python.Python.3.12` |
| [Ollama](https://ollama.com) | Local AI inference | macOS `brew install ollama`, Windows `winget install Ollama.Ollama`, or download from ollama.com |
| Rust toolchain | Builds the screen-capture helper from source (see below) | [rustup.rs](https://rustup.rs) |
| macOS: Xcode Command Line Tools | Needed to compile the Rust helper | `xcode-select --install` |
| Windows: Visual Studio Build Tools (C++ workload) | Needed to compile the Rust helper (MSVC linker) | [visualstudio.microsoft.com/downloads](https://visualstudio.microsoft.com/downloads/) |

The screen-capture helper (`native_capture/rust/`) is included here as
source, not a prebuilt binary -- you compile it yourself with `cargo
build`, so you can read exactly what it does before trusting it with
Screen Recording/Accessibility/Input Monitoring permissions.

## Setup

### 1. Get the code running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### 2. Start Ollama and pull a model

```bash
ollama serve
```

In another terminal:

```bash
ollama pull qwen3-coder-next:latest
curl http://localhost:11434/api/tags   # sanity check
```

Using a different model? Set `OLLAMA_MODEL` in `.env` to match. Bigger
models are more capable but slower to respond.

Optionally, also pull a **vision-capable** model for vision-assisted
evaluation at Finish time (looking at actual screenshots, not just OCR'd
text):

```bash
ollama pull llava
```

Set `OLLAMA_VISION_MODEL` in `.env` to use a different multimodal model,
or leave it empty to disable vision-assisted evaluation (evaluation still
works fine on text evidence alone).

### 3. Build the screen-capture helper

```bash
cd native_capture/rust
cargo build
cd ../..
```

Then check permission status:

```bash
native_capture/rust/target/debug/cyberalfred-capture check
```

On macOS this needs **three** permissions -- grant them to your terminal
app in System Settings -> Privacy & Security:

- **Screen Recording** -- for screenshots
- **Accessibility** -- for on-screen text extraction and window titles
- **Input Monitoring** -- for mouse/keyboard activity capture (never raw
  keystrokes, only categorized activity counts)

Re-run `check` until all three say "granted". Capture still runs in
degraded mode with permissions missing -- exercises just won't have full
evidence until you grant them.

Then point the app at the helper you just built, in `.env`:

```
EVIDENCE_PROVIDER=rust
NATIVE_RUST_CAPTURE_EXECUTABLE=./native_capture/rust/target/debug/cyberalfred-capture
```

See `native_capture/rust/README.md` for exactly what this helper captures,
its full CLI surface, and its privacy notes.

### 4. Start the app

```bash
source .venv/bin/activate      # macOS
python run.py
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python run.py
```

Open `http://127.0.0.1:8080`. The startup banner tells you whether Ollama
is reachable and which model is configured.

> **Restarting after changes:** the app does not hot-reload. If you edit
> `.env` or any Python file, stop the server (Ctrl+C) and run
> `python run.py` again.

## Using the App

1. **Import an exercise** if you received a `.yaml` file: on the home
   page, open **Import an Exercise**, choose the file, and click Import.
2. **Start an exercise**: click an exercise card, review the instructions,
   click **Start Exercise**. Do the work in your actual applications (e.g.
   Terminal) -- the app doesn't embed a terminal, it just watches via the
   capture helper you set up above, starting it automatically for the
   session.

   Each exercise shows its **Difficulty**. If it shows **Open**, you'll
   see a dropdown to pick your own difficulty before starting; that choice
   applies to this session only and shapes how much help the mentor gives.
3. **Ask the mentor questions** any time from the session page (e.g. "Am
   I doing this correctly?"). It only answers based on what it can
   actually observe or verify.
4. **Finish the exercise** when done. You'll get a score, a breakdown per
   expected outcome, and narrative feedback (strengths, improvements,
   risky/unnecessary steps, alternative approaches).
5. **Share your result** with whoever's collecting beta feedback via the
   **Download Result to Share** button on the results page -- a signed
   JSON file you send however you like.

You can run the same exercise as many times as you want; each attempt is
a separate session.

## Reporting bugs

Please include: what you were doing, what you expected, what happened
instead, and (if relevant) anything printed in the terminal running
`python run.py`.
