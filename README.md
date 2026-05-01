# pi-ollama-provider

Ollama provider extension for [pi](https://github.com/badlogic/pi-mono) — auto-discovers local and cloud models, detects capabilities, starts fast from cache, and auto-pulls on demand.

## Capabilities

- **Local and cloud models** — discovers both local and cloud models from Ollama
- **Auto-discovery** — fetches all models from `localhost:11434/api/tags` and registers them as pi provider models on startup
- **Persistent cache** — writes discovered models to `~/.pi/agent/ollama-models-cache.json` for instant availability on next startup; blocks on first run (no cache) so models are ready before the session starts
- **Vision detection** — marks models with `input: ["text", "image"]` when they support vision, detected via:
  - `capabilities: ["vision"]` (Ollama ≥ 0.6)
  - Known vision architectures (`llava`, `mllama`, `minicpm-v`, `phi3-v`, etc.)
  - `clip.has_vision_encoder` flag
- **Reasoning detection** — marks `reasoning: true` when Ollama reports `capabilities: ["thinking"]`
- **Accurate context windows** — reads `.context_length` from `/api/show` for each model; falls back to 32K for local, 128K for cloud models
- **Cloud model support** — models tagged `:cloud` are identified from local Ollama and displayed with a cloud label
- **Auto-pull on select** — when you select an Ollama model that isn't installed locally, it auto-pulls with a streaming progress bar
- **`/ollama-setup`** — interactive setup wizard: choose local or cloud, install Ollama, authenticate
- **`/ollama-pull <model>`** — manually pull a model with progress bar
- **`/ollama-refresh`** — re-discover models (e.g., after pulling from the CLI)
- **Cloud direct connect** — connect directly to ollama.com with an API key (no local install needed)
- **Single source of truth** — only uses the Ollama `/api/tags` endpoint (local or cloud)

## Install

```bash
pi install git:github.com/v2nic/pi-ollama-provider
```

Or in `settings.json`:

```json
{
  "packages": ["git:github.com/v2nic/pi-ollama-provider"]
}
```

## Setup

Run the interactive setup wizard:

```
/ollama-setup
```

The wizard guides you through:

1. **Local or Cloud** — choose whether to run models locally or connect to ollama.com
2. **Local mode:**
   - Installs Ollama if not found (`curl -fsSL https://ollama.com/install.sh | sh`)
   - Starts the Ollama service if not running
   - Optionally configures cloud model access via `ollama signin` or API key
3. **Cloud mode:**
   - Asks for your ollama.com API key
   - Configures the extension to connect directly (no local Ollama required)

Configuration is saved to `~/.pi/agent/auth.json` under the `ollama` key (shared with all pi providers).
You can also set the `OLLAMA_API_KEY` environment variable — it takes effect automatically
without running the wizard.

### Environment variable

Set `OLLAMA_API_KEY` to connect to ollama.com without the wizard:

```bash
export OLLAMA_API_KEY=your-key-here
```

### Manual authentication (optional)

[Authenticate](https://docs.ollama.com/api/authentication) with Ollama Cloud to use cloud models:

```bash
ollama signin
```

The first time you run `pi`, it will print

`[ollama] 22 models discovered`

Then on subsequent runs, it will use the cache and print:

`[ollama] 22 models from cache`

## Commands

| Command | Description |
|---------|-------------|
| `/ollama-setup` | Interactive setup wizard (local or cloud) |
| `/models` | Ollama models are available in the model list |
| `/ollama-refresh` | Re-discover models from Ollama |
| `/ollama-pull <model>` | Pull a model with streaming progress bar |

## How It Works

1. On startup, registers models from cache (synchronous, instant)
2. Fetches `/api/tags` from local Ollama and `/api/show` for each model (async, 5s timeout)
3. Detects vision, reasoning, cloud, and context window from the responses
4. Updates the provider and writes cache for next startup
5. If no cache exists, returns a promise so pi waits for models before resolving the configured model

## Comparison with [pi-ollama](https://github.com/0xKobold/pi-ollama)

| | **pi-ollama-provider** | **pi-ollama** |
|---|---|---|
| Model source | Local `/api/tags` only | Local `/api/tags` + cloud API via `ollama-js` |
| Cloud config | Shared `auth.json` + env var (`OLLAMA_API_KEY`) | Settings files + env vars (`OLLAMA_HOST`, `OLLAMA_API_KEY`) |
| Cloud models | `:cloud` tag detected from local Ollama | Separate `ollama-cloud` provider with API key |
| Vision detection | ✅ capabilities + architecture + CLIP flag | ✅ capabilities + architecture + CLIP flag |
| Reasoning detection | `capabilities: ["thinking"]` from `/api/show` | Heuristic from model name (`r1`, `deepseek`, etc.) |
| Context window | From `/api/show` `.context_length`, fallback 32K | From `/api/show` + heuristic by model name/size |
| Extra commands | `/ollama-refresh`, `/ollama-pull` | `/ollama-status`, `/ollama-info`, `/ollama-models` |
| Auto-pull | ✅ On model select, with progress bar | ❌ None |
| Pull command | `/ollama-pull <model>` | ❌ None |
| Startup | Cache-first (sync), blocks on first run | Always async fetch, no cache |
| Dependencies | Zero — raw `fetch()` | `ollama` npm package, `@sinclair/typebox` |
| Lines of code | ~300 | ~500+ |
