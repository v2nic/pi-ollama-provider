# pi-ollama-provider

Ollama provider extension for [pi](https://github.com/badlogic/pi-mono) — native `/api/chat` streaming, auto-discovers local and cloud models, detects capabilities, starts fast from cache, auto-pulls on demand, and implements all recommendations from [the Ollama integration research document](https://github.com/v2nic/pi-ollama-provider/issues/6).

## What's New in v2.0

This version implements the full set of recommendations from the [Ollama Provider Integration Research](ollama-provider-integration-research.md):

| Feature | v1.0 | v2.0 |
|---------|------|------|
| **Streaming** | OpenAI compat only (broken tool calls) | ✅ **Native `/api/chat`** with OpenAI compat fallback |
| **Tool calling** | ❌ Silently dropped in streaming (ollama#12557) | ✅ Works correctly via native API |
| **`num_ctx`** | ❌ Not set (Ollama defaults to 4096) | ✅ Always set from model data (prevents silent truncation) |
| **Ghost-token retry** | ❌ None | ✅ Detects empty streams and retries with `stream: false` |
| **Context overflow** | ❌ Not detected | ✅ Recognizes Ollama 400 errors |
| **Tool capability detection** | ❌ None | ✅ From `/api/show` capabilities + family heuristics |
| **Separate cloud provider** | ❌ Single `ollama` provider | ✅ `ollama` + `ollama-cloud` providers |
| **`/ollama-status`** | ❌ | ✅ Connection status + loaded models |
| **`/ollama-info`** | ❌ | ✅ Detailed model info from `/api/show` |
| **Sampling settings** | ❌ None | ✅ `temperature`, `top_p`, `top_k` via settings.json |
| **`OLLAMA_HOST`** | ❌ Hardcoded localhost | ✅ Respects env var |
| **Web tools (cloud)** | ❌ | ✅ `ollama_web_search` + `ollama_web_fetch` |
| **`OLLAMA_KEEP_ALIVE`** | ❌ Default 5 min | ✅ Default 30 min for agent sessions |
| **Cloud model filtering** | ❌ All models | ✅ Only tool-capable models |

## Capabilities

- **Native /api/chat streaming** — Fixes the tool-calling bug (ollama#12557) where the OpenAI shim silently drops tool_calls in streaming mode
- **Always sets `num_ctx`** — Prevents Ollama's 4096 default from silently truncating system prompts and conversation history
- **Ghost-token retry** — Detects when the model generated tokens but streaming returned nothing, automatically retries with `stream: false`
- **Context overflow detection** — Recognizes Ollama's 400 errors ("prompt too long; exceeded max context length") for proper auto-compaction
- **Local and cloud models** — Discovers models from local Ollama at `localhost:11434` and cloud models at `ollama.com`
- **Two-tier model discovery** — Cache-first registration (instant startup), then background refresh from `/api/tags` + `/api/show`
- **Tool capability detection** — Detects from `capabilities: ["tools"]` and known tool-capable families (llama3.1+, qwen2.5+, gemma4, etc.)
- **Vision detection** — Marks models with `input: ["text", "image"]` via capabilities, architecture, and CLIP flags
- **Reasoning detection** — Detects thinking mode from capabilities and model name patterns
- **Accurate context windows** — Reads `.context_length` from `/api/show`; falls back to 32K for local, 128K for cloud
- **Separate cloud provider** — `ollama-cloud` provider connects directly to ollama.com/v1 (no local server needed)
- **Cloud web tools** — `ollama_web_search` and `ollama_web_fetch` tools for Ollama Cloud (enable via `PI_OLLAMA_WEB_TOOLS=1`)
- **Auto-pull on select** — When you select a model that isn't installed, it auto-pulls with a streaming progress bar
- **`OLLAMA_HOST` support** — Respects the `OLLAMA_HOST` environment variable for custom server locations
- **`OLLAMA_KEEP_ALIVE`** — Defaults to 30 minutes for agent sessions (prevents model unloading between turns)
- **Sampling parameters** — Configure `temperature`, `top_p`, `top_k`, etc. via `~/.pi/agent/settings.json`
- **`/ollama-setup`** — Interactive setup wizard: choose local or cloud, install Ollama, authenticate
- **`/ollama-pull <model>`** — Manually pull a model with progress bar
- **`/ollama-refresh`** — Re-discover models (e.g., after pulling from the CLI)
- **`/ollama-status`** — Check Ollama connection status and loaded models
- **`/ollama-info <model>`** — Show detailed model info from `/api/show`
- **Single source of truth** — Only uses Ollama's `/api/tags`, `/api/show`, and `/api/ps` endpoints
- **Zero dependencies** — Pure TypeScript, no npm packages, only raw `fetch()`

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

Set `OLLAMA_HOST` to use a custom Ollama server:

```bash
export OLLAMA_HOST=http://192.168.1.100:11434
```

Set `OLLAMA_KEEP_ALIVE` to control how long models stay loaded (default: 30m):

```bash
export OLLAMA_KEEP_ALIVE=1h
```

### Settings

Configure Ollama-specific options in `~/.pi/agent/settings.json`:

```json
{
  "ollama": {
    "streamingMode": "native",
    "keepAlive": "30m",
    "autoPull": true,
    "options": {
      "temperature": 0.7,
      "top_p": 0.9,
      "num_predict": 4096
    }
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `streamingMode` | `"native"` | Use native `/api/chat` or OpenAI compat `/v1/chat/completions` |
| `keepAlive` | `"30m"` | How long models stay loaded after last request |
| `autoPull` | `true` | Auto-pull models on select |
| `defaultNumCtx` | model's context_length | Override num_ctx for all models |
| `options.temperature` | model default | Sampling temperature |
| `options.top_p` | model default | Nucleus sampling threshold |
| `options.top_k` | model default | Top-K sampling |
| `options.num_predict` | model default | Max tokens to generate |

### Web tools (cloud)

To enable `ollama_web_search` and `ollama_web_fetch` tools:

```bash
export PI_OLLAMA_WEB_TOOLS=1
```

These tools use Ollama Cloud's hosted web search/fetch endpoints, which are only available for cloud models.

## Commands

| Command | Description |
|---------|-------------|
| `/ollama-setup` | Interactive setup wizard (local or cloud) |
| `/ollama-refresh` | Re-discover models from Ollama |
| `/ollama-pull <model>` | Pull a model with streaming progress bar |
| `/ollama-status` | Check connection status, loaded models, settings |
| `/ollama-info <model>` | Show detailed model info from /api/show |

## How It Works

1. On startup, registers models from cache (synchronous, instant)
2. Fetches `/api/tags` from local Ollama and `/api/show` for each model (async, 5s timeout)
3. Detects tools, vision, reasoning, and context window from the responses
4. Registers provider with `streamSimple` using native `/api/chat` streaming
5. Updates the provider and writes cache for next startup
6. If no cache exists, returns a promise so pi waits for models before resolving the configured model
7. For cloud: fetches `/v1/models` from ollama.com, filters to tool-capable models only

### Native Streaming Architecture

```
pi agent loop
     │
     ▼
streamSimple (registered in pi.registerProvider)
     │
     ▼
POST /api/chat (native Ollama NDJSON)
     │
     ├── Text deltas → stream.pushTextDelta()
     ├── Thinking deltas → stream.pushThinkingDelta()
     ├── Tool calls → stream.pushToolCall()
     ├── Usage data → stream.pushUsage()
     ├── Ghost token? → Retry with stream: false
     └── Overflow error? → Throw with context info
```

This fixes the critical tool-calling bug (ollama#12557) where the OpenAI compat shim silently drops `tool_calls` in streaming mode. The native `/api/chat` endpoint delivers tool calls correctly as a complete burst.

### Context Safety

The extension always sets `num_ctx` in every request based on the model's `context_length` from `/api/show`. This prevents the #1 pitfall for coding agents using Ollama: the silent 4096-token truncation that causes lost system prompts and forgotten conversation context (issues #133, #2177, #2794, #3859).

## Architecture

```
extensions/pi-ollama-provider/
├── index.ts           # Extension entry point + provider registration
├── auth.ts            # Auth resolution (auth.json + env vars)
├── native-stream.ts   # Native /api/chat NDJSON streaming + tool calls
├── discovery.ts       # Model discovery + capability inference + cache
├── cloud.ts           # Cloud provider + web search/fetch tools
├── context-safety.ts   # num_ctx, overflow detection, retry config
└── commands.ts        # Slash commands, settings, setup wizard
```

## Testing

```bash
npm test          # Run all tests
npm run test:watch # Watch mode
```

116 tests across 6 test files covering:
- Auth resolution priority chain
- NDJSON parsing (single/multi/chunked/malformed)
- Message conversion (developer→system, images, tool results)
- Tool definition conversion
- Ghost-token detection
- Context overflow detection
- Capability inference (vision, tools, reasoning)
- Context length extraction
- Cloud model detection
- OLLAMA_HOST env var handling
- num_ctx calculation
- Setup wizard flows

## Comparison with Community Extensions

| | **pi-ollama-provider v2** | **[pi-ollama](https://github.com/0xKobold/pi-ollama)** | **[pi-ollama-native](https://github.com/CaptCanadaMan/pi-ollama-native)** |
|---|---|---|---|
| Streaming | Native /api/chat + compat fallback | OpenAI compat only | Native /api/chat (core patch) |
| Tool calling | ✅ Works (native API) | ❌ Broken (ollama#12557) | ✅ Works (core patch) |
| Context safety | ✅ Always sets num_ctx | ❌ Not set | ✅ Sets num_ctx |
| Ghost-token retry | ✅ Automatic | ❌ None | ✅ Manual detection |
| Overflow detection | ✅ | ❌ | ❌ |
| Cloud provider | ✅ Separate (ollama-cloud) | ✅ Separate | ❌ Local only |
| Web tools | ✅ ollama_web_search/fetch | ❌ | ❌ |
| Core changes needed | ❌ Extension only | ❌ Extension only | ⚠️ 7 edits to 6 core files |
| OLLAMA_HOST | ✅ | ✅ | ❌ |
| Auto-pull | ✅ With progress | ❌ | ❌ |
| Dependencies | Zero | ollama npm, typebox | None (but core patches) |

## License

MIT