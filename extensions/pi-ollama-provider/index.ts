/**
 * Ollama Auto-Discovery Extension
 *
 * Discovers models from the local Ollama instance and registers them
 * via pi.registerProvider(). Cloud models (:cloud tag) are marked accordingly.
 * Non-installed models are auto-pulled on first use with progress bar.
 *
 * Commands:
 * - /ollama-setup: interactive setup wizard for local or cloud Ollama
 * - /ollama-refresh: re-discovers models
 * - /ollama-pull <model>: pull a model with progress bar
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

const PROVIDER_NAME = "ollama";
const DEFAULT_CONTEXT_WINDOW = 32768;
const DEFAULT_MAX_TOKENS = 32768;
const CONFIG_DIR = join(homedir(), ".pi", "agent");
const CACHE_PATH = join(CONFIG_DIR, "ollama-models-cache.json");
const CONFIG_PATH = join(CONFIG_DIR, "ollama-config.json");

const DEFAULT_LOCAL_URL = "http://localhost:11434";
const DEFAULT_CLOUD_URL = "https://api.ollama.com";

let localModelNames = new Set<string>();
let pullingModels = new Set<string>();

// ── config ──

interface OllamaConfig {
  mode: "local" | "cloud";
  baseUrl: string;
  apiKey?: string;
}

function getDefaultConfig(): OllamaConfig {
  return { mode: "local", baseUrl: DEFAULT_LOCAL_URL };
}

function readConfig(): OllamaConfig {
  try {
    const data = readFileSync(CONFIG_PATH, "utf-8");
    const parsed = JSON.parse(data);
    if (parsed && typeof parsed === "object" && parsed.mode && parsed.baseUrl) {
      return parsed as OllamaConfig;
    }
  } catch {}
  return getDefaultConfig();
}

function writeConfig(config: OllamaConfig): void {
  try {
    mkdirSync(CONFIG_DIR, { recursive: true });
    writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2), "utf-8");
  } catch (err) {
    console.log(`[ollama] config write failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

let currentConfig: OllamaConfig = readConfig();

function getBaseUrl(): string {
  return currentConfig.baseUrl;
}

function getApiKey(): string {
  return currentConfig.apiKey || "ollama";
}

// ── cache ──

function readModelCache(): any[] | null {
  try {
    const data = readFileSync(CACHE_PATH, "utf-8");
    const parsed = JSON.parse(data);
    return Array.isArray(parsed) && parsed.length > 0 ? parsed : null;
  } catch {
    return null;
  }
}

function writeModelCache(models: any[]): void {
  try {
    mkdirSync(CONFIG_DIR, { recursive: true });
    writeFileSync(CACHE_PATH, JSON.stringify(models, null, 2), "utf-8");
  } catch (err) {
    console.log(`[ollama] cache write failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

function registerFromCache(pi: ExtensionAPI): boolean {
  const cached = readModelCache();
  if (!cached) return false;
  pi.unregisterProvider(PROVIDER_NAME);
  pi.registerProvider(PROVIDER_NAME, {
    baseUrl: `${getBaseUrl()}/v1`,
    api: "openai-completions",
    apiKey: getApiKey(),
    compat: { supportsDeveloperRole: false, supportsReasoningEffort: false },
    models: cached,
  });
  console.log(`[ollama] ${cached.length} models from cache`);
  return true;
}

// ── ollama api ──

interface ModelEntry {
  name: string;
  size: number;
  details?: { capabilities?: string[] };
  model_info?: Record<string, unknown>;
}

async function fetchLocalModels(): Promise<ModelEntry[]> {
  const baseUrl = getBaseUrl();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (currentConfig.apiKey) {
    headers["Authorization"] = `Bearer ${currentConfig.apiKey}`;
  }
  try {
    const res = await fetch(`${baseUrl}/api/tags`, { headers });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const models: ModelEntry[] = data.models || [];
    localModelNames = new Set(models.map((m) => m.name));
    return models;
  } catch (err) {
    console.log(`[ollama] unavailable (${baseUrl}): ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

async function fetchModelDetails(name: string): Promise<ModelEntry | null> {
  const baseUrl = getBaseUrl();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (currentConfig.apiKey) {
    headers["Authorization"] = `Bearer ${currentConfig.apiKey}`;
  }
  try {
    const res = await fetch(`${baseUrl}/api/show`, {
      method: "POST",
      headers,
      body: JSON.stringify({ name }),
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

// ── capabilities ──

/** Detect vision support from /api/show details and /api/tags metadata. */
function hasVision(
  capabilities: string[],
  modelInfo: Record<string, unknown>,
): boolean {
  if (capabilities.some((c) => c === "vision")) return true;
  const arch = String(modelInfo["general.architecture"] ?? "").toLowerCase();
  if (["llava", "bakllava", "moondream", "llava-next", "minicpm-v", "phi3-v", "mllama"].some((v) => arch.includes(v))) return true;
  if (modelInfo["clip.has_vision_encoder"] === true) return true;
  return false;
}

// ── pull ──

async function pullModelWithProgress(modelName: string, ctx?: any): Promise<void> {
  if (pullingModels.has(modelName)) {
    while (pullingModels.has(modelName)) await new Promise((r) => setTimeout(r, 500));
    return;
  }
  pullingModels.add(modelName);

  const baseUrl = getBaseUrl();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (currentConfig.apiKey) {
    headers["Authorization"] = `Bearer ${currentConfig.apiKey}`;
  }

  try {
    const res = await fetch(`${baseUrl}/api/tags`, { headers });
    if (res.ok) {
      const data = await res.json();
      if ((data.models || []).some((m: any) => m.name === modelName)) {
        localModelNames.add(modelName);
        return;
      }
    }

    const progressId = `ollama-pull-${modelName.replace(/[^a-z0-9]/gi, "-")}`;
    ctx?.ui?.setStatus(progressId, `⬇ Pulling ${modelName}...`);

    try {
      const pullRes = await fetch(`${baseUrl}/api/pull`, {
        method: "POST",
        headers,
        body: JSON.stringify({ name: modelName, stream: true }),
      });

      if (!pullRes.ok && !pullRes.body) throw new Error(`HTTP ${pullRes.status}`);

      const reader = pullRes.body!.getReader();
      const decoder = new TextDecoder();
      let lastProgress = 0;
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop()!;

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const event = JSON.parse(line);
            if (event.error) {
              throw new Error(event.error);
            } else if (event.status === "success") {
              lastProgress = 100;
              ctx?.ui?.setStatus(progressId, `✓ ${modelName} ready!`);
            } else if (event.total && event.completed && event.total > 0) {
              const pct = Math.round((event.completed / event.total) * 100);
              if (pct > lastProgress) {
                lastProgress = pct;
                const filled = Math.floor(pct / 5);
                ctx?.ui?.setStatus(
                  progressId,
                  `⬇ [${"█".repeat(filled)}${"░".repeat(20 - filled)}] ${pct}% ${modelName}`,
                );
              }
            } else if (event.status && lastProgress === 0) {
              ctx?.ui?.setStatus(progressId, `⬇ ${event.status} ${modelName}`);
            }
          } catch (e) {
            if (e instanceof Error) throw e;
          }
        }
      }

      localModelNames.add(modelName);
      ctx?.ui?.setStatus(progressId, `✓ ${modelName} ready!`);
      setTimeout(() => ctx?.ui?.setStatus(progressId, undefined), 4000);
    } catch (err) {
      ctx?.ui?.setStatus(
        progressId,
        `✗ ${modelName}: ${err instanceof Error ? err.message : String(err)}`,
      );
      setTimeout(() => ctx?.ui?.setStatus(progressId, undefined), 5000);
      throw err;
    }
  } finally {
    pullingModels.delete(modelName);
  }
}

// ── register ──

async function registerOllamaProvider(pi: ExtensionAPI): Promise<void> {
  const models = await fetchLocalModels();
  if (models.length === 0) return;

  const detailResults = new Map<string, ModelEntry | null>();
  await Promise.race([
    Promise.all(
      models.map(async (m) => {
        detailResults.set(m.name, await fetchModelDetails(m.name));
      }),
    ),
    new Promise((r) => setTimeout(r, 5000)),
  ]);

  console.log(`[ollama] ${models.length} models discovered`);

  const modelConfigs = models.map((model) => {
    const details = detailResults.get(model.name);
    const capabilities: string[] = details?.details?.capabilities || model.details?.capabilities || [];
    const modelInfo: Record<string, unknown> = { ...(model.model_info || {}), ...(details?.model_info || {}) };

    let contextWindow = DEFAULT_CONTEXT_WINDOW;
    let maxTokens = DEFAULT_MAX_TOKENS;

    const ctxKeys = Object.keys(modelInfo).filter((k) => k.endsWith(".context_length"));
    if (ctxKeys.length > 0) {
      const val = Number(modelInfo[ctxKeys[0]]);
      if (Number.isFinite(val) && val > 0) {
        contextWindow = val;
        maxTokens = Math.min(val * 4, 131072);
      }
    }

    const isCloud = model.name.includes(":cloud") || (!localModelNames.has(model.name) && model.size > 100_000_000_000);
    if (isCloud && contextWindow === DEFAULT_CONTEXT_WINDOW) {
      contextWindow = 131072;
      maxTokens = 131072;
    }

    const isVision = hasVision(capabilities, modelInfo);

    return {
      id: model.name,
      name: isCloud ? `${model.name} (cloud)` : model.name,
      reasoning: capabilities.includes("thinking"),
      input: isVision ? ["text", "image"] as const : ["text"] as const,
      contextWindow,
      maxTokens,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    };
  });

  pi.unregisterProvider(PROVIDER_NAME);
  pi.registerProvider(PROVIDER_NAME, {
    baseUrl: `${getBaseUrl()}/v1`,
    api: "openai-completions",
    apiKey: getApiKey(),
    compat: { supportsDeveloperRole: false, supportsReasoningEffort: false },
    models: modelConfigs,
  });

  writeModelCache(modelConfigs);
}

// ── setup wizard ──

async function checkOllamaInstalled(pi: ExtensionAPI): Promise<boolean> {
  try {
    const result = await pi.exec("ollama", ["--version"], { timeout: 5000 });
    return result.code === 0;
  } catch {
    return false;
  }
}

async function checkOllamaRunning(): Promise<boolean> {
  try {
    const res = await fetch(`${DEFAULT_LOCAL_URL}/api/tags`);
    return res.ok;
  } catch {
    return false;
  }
}

async function installOllama(pi: ExtensionAPI, ctx: any): Promise<boolean> {
  ctx.ui.setStatus("ollama-setup", "⬇ Installing Ollama...");
  try {
    const result = await pi.exec("bash", ["-c", "curl -fsSL https://ollama.com/install.sh | sh"], { timeout: 120000 });
    ctx.ui.setStatus("ollama-setup", undefined);
    if (result.code === 0) {
      ctx.ui.notify("✓ Ollama installed successfully!", "info");
      return true;
    } else {
      ctx.ui.notify(`✗ Installation failed: ${result.stderr}`, "error");
      return false;
    }
  } catch (err) {
    ctx.ui.setStatus("ollama-setup", undefined);
    ctx.ui.notify(`✗ Installation failed: ${err instanceof Error ? err.message : String(err)}`, "error");
    return false;
  }
}

async function startOllamaService(pi: ExtensionAPI, ctx: any): Promise<boolean> {
  ctx.ui.setStatus("ollama-setup", "Starting Ollama...");
  try {
    // Start ollama serve in background
    pi.exec("bash", ["-c", "nohup ollama serve > /dev/null 2>&1 &"], { timeout: 5000 }).catch(() => {});
    // Wait for it to be ready
    for (let i = 0; i < 10; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      if (await checkOllamaRunning()) {
        ctx.ui.setStatus("ollama-setup", undefined);
        ctx.ui.notify("✓ Ollama is running!", "info");
        return true;
      }
    }
    ctx.ui.setStatus("ollama-setup", undefined);
    ctx.ui.notify("✗ Ollama did not start in time. Try running 'ollama serve' manually.", "error");
    return false;
  } catch (err) {
    ctx.ui.setStatus("ollama-setup", undefined);
    ctx.ui.notify(`✗ Could not start Ollama: ${err instanceof Error ? err.message : String(err)}`, "error");
    return false;
  }
}

async function runSetupWizard(pi: ExtensionAPI, ctx: any): Promise<void> {
  // Step 1: Choose connection mode
  const mode = await ctx.ui.select(
    "🦙 Ollama Setup — How would you like to connect?",
    [
      "Local — Run models on this machine",
      "Cloud — Connect to ollama.com (API key required)",
    ],
  );

  if (!mode) return; // cancelled

  if (mode.startsWith("Cloud")) {
    // ── Cloud setup ──
    const apiKey = await ctx.ui.input(
      "Enter your Ollama API key (from ollama.com/settings/keys):",
      "",
    );

    if (!apiKey) {
      ctx.ui.notify("Setup cancelled — no API key provided.", "warning");
      return;
    }

    // Optionally let user customize the endpoint
    const customUrl = await ctx.ui.input(
      "Ollama cloud endpoint (press Enter for default):",
      DEFAULT_CLOUD_URL,
    );

    const baseUrl = customUrl?.trim() || DEFAULT_CLOUD_URL;

    // Test the connection
    ctx.ui.setStatus("ollama-setup", "Testing cloud connection...");
    try {
      const res = await fetch(`${baseUrl}/api/tags`, {
        headers: {
          "Authorization": `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
      });

      ctx.ui.setStatus("ollama-setup", undefined);

      if (!res.ok) {
        const body = await res.text().catch(() => "");
        ctx.ui.notify(`✗ Connection failed (HTTP ${res.status}): ${body}`, "error");
        return;
      }

      const data = await res.json();
      const modelCount = (data.models || []).length;

      // Save config
      currentConfig = { mode: "cloud", baseUrl, apiKey };
      writeConfig(currentConfig);

      ctx.ui.notify(`✓ Connected to Ollama Cloud! ${modelCount} models available.`, "info");

      // Refresh models
      await registerOllamaProvider(pi);
    } catch (err) {
      ctx.ui.setStatus("ollama-setup", undefined);
      ctx.ui.notify(`✗ Connection failed: ${err instanceof Error ? err.message : String(err)}`, "error");
    }
    return;
  }

  // ── Local setup ──
  const isInstalled = await checkOllamaInstalled(pi);

  if (!isInstalled) {
    const install = await ctx.ui.confirm(
      "Ollama not found",
      "Ollama is not installed on this machine. Install it now?",
    );

    if (!install) {
      ctx.ui.notify("Setup cancelled. Install Ollama manually: https://ollama.com/download", "warning");
      return;
    }

    const installed = await installOllama(pi, ctx);
    if (!installed) return;
  }

  // Check if Ollama is running
  const isRunning = await checkOllamaRunning();
  if (!isRunning) {
    const start = await ctx.ui.confirm(
      "Ollama not running",
      "Ollama is installed but not running. Start it now?",
    );

    if (start) {
      const started = await startOllamaService(pi, ctx);
      if (!started) return;
    } else {
      ctx.ui.notify("Start Ollama manually with 'ollama serve' before using models.", "warning");
      return;
    }
  }

  // Ask about cloud model access
  const wantCloud = await ctx.ui.confirm(
    "Cloud models",
    "Would you like to access Ollama cloud models (e.g. GPT, Claude via Ollama)?\nThis requires an Ollama account.",
  );

  if (wantCloud) {
    const loginMethod = await ctx.ui.select(
      "How would you like to authenticate for cloud models?",
      [
        "Browser login (ollama signin)",
        "API key",
      ],
    );

    if (!loginMethod) {
      // Still save local config
      currentConfig = { mode: "local", baseUrl: DEFAULT_LOCAL_URL };
      writeConfig(currentConfig);
      await registerOllamaProvider(pi);
      ctx.ui.notify("✓ Local Ollama configured (no cloud models).", "info");
      return;
    }

    if (loginMethod.startsWith("Browser")) {
      ctx.ui.notify("Running 'ollama signin'... Follow the browser prompts.", "info");
      ctx.ui.setStatus("ollama-setup", "Waiting for ollama signin...");
      try {
        const result = await pi.exec("ollama", ["signin"], { timeout: 120000 });
        ctx.ui.setStatus("ollama-setup", undefined);
        if (result.code === 0) {
          ctx.ui.notify("✓ Signed in to Ollama! Cloud models should now be available.", "info");
        } else {
          ctx.ui.notify(`Sign-in output: ${result.stderr || result.stdout}`, "warning");
        }
      } catch (err) {
        ctx.ui.setStatus("ollama-setup", undefined);
        ctx.ui.notify(`Sign-in error: ${err instanceof Error ? err.message : String(err)}`, "error");
      }

      // Save local config (cloud access is via ollama signin, not API key)
      currentConfig = { mode: "local", baseUrl: DEFAULT_LOCAL_URL };
      writeConfig(currentConfig);
    } else {
      // API key flow for local+cloud
      const apiKey = await ctx.ui.input(
        "Enter your Ollama API key:",
        "",
      );

      if (!apiKey) {
        currentConfig = { mode: "local", baseUrl: DEFAULT_LOCAL_URL };
        writeConfig(currentConfig);
        ctx.ui.notify("✓ Local Ollama configured (no cloud API key).", "info");
        await registerOllamaProvider(pi);
        return;
      }

      currentConfig = { mode: "local", baseUrl: DEFAULT_LOCAL_URL, apiKey };
      writeConfig(currentConfig);
      ctx.ui.notify("✓ Local Ollama configured with API key for cloud models.", "info");
    }
  } else {
    // Pure local
    currentConfig = { mode: "local", baseUrl: DEFAULT_LOCAL_URL };
    writeConfig(currentConfig);
  }

  // Refresh models with new config
  await registerOllamaProvider(pi);
  ctx.ui.notify("✓ Setup complete! Use /models to see available Ollama models.", "info");
}

// ── entry ──

export default function (pi: ExtensionAPI) {
  // Load config
  currentConfig = readConfig();

  // Always register commands first, so they're available even on first run
  pi.registerCommand("ollama-setup", {
    description: "Interactive setup wizard for Ollama (local or cloud)",
    handler: async (_args, ctx) => {
      await runSetupWizard(pi, ctx);
    },
  });

  pi.registerCommand("ollama-refresh", {
    description: "Re-discover Ollama models",
    handler: async (_args, ctx) => {
      ctx.ui.notify("[ollama] Discovering models...", "info");
      await registerOllamaProvider(pi);
    },
  });

  pi.registerCommand("ollama-pull", {
    description: "Pull an Ollama model with progress bar",
    handler: async (args, ctx) => {
      const name = typeof args === "string" ? args.trim() : args?.model || args?.[0];
      if (!name) {
        ctx.ui.notify("Usage: /ollama-pull <model-name>", "error");
        return;
      }
      try {
        await pullModelWithProgress(name, ctx);
      } catch (err) {
        ctx.ui.notify(`Pull failed: ${err instanceof Error ? err.message : String(err)}`, "error");
      }
    },
  });

  pi.on("model_select", async (event) => {
    if (event.model?.provider !== PROVIDER_NAME) return;
    const modelId = event.model?.id;
    if (!modelId || localModelNames.has(modelId)) return;
    try {
      await pullModelWithProgress(modelId, event);
    } catch (err) {
      console.error(`[ollama] auto-pull failed: ${modelId}: ${err}`);
    }
  });

  // Register from cache synchronously (instant), then refresh from Ollama.
  const hasCache = registerFromCache(pi);

  const ready = registerOllamaProvider(pi);
  if (!hasCache) {
    // No cache — block so models are available before session starts
    return ready;
  }
}
