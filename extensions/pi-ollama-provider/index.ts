/**
 * Ollama Auto-Discovery Extension
 *
 * Discovers models from the local Ollama instance and registers them
 * via pi.registerProvider(). Cloud models (:cloud tag) are marked accordingly.
 * Non-installed models are auto-pulled on first use with progress bar.
 *
 * Commands:
 * - /ollama-refresh: re-discovers models
 * - /ollama-pull <model>: pull a model with progress bar
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

const OLLAMA_BASE_URL = "http://localhost:11434";
const PROVIDER_NAME = "ollama";
const DEFAULT_CONTEXT_WINDOW = 32768;
const DEFAULT_MAX_TOKENS = 32768;
const CACHE_PATH = join(homedir(), ".pi", "agent", "ollama-models-cache.json");

let localModelNames = new Set<string>();
let pullingModels = new Set<string>();

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
    mkdirSync(join(CACHE_PATH, ".."), { recursive: true });
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
    baseUrl: `${OLLAMA_BASE_URL}/v1`,
    api: "openai-completions",
    apiKey: "ollama",
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
  try {
    const res = await fetch(`${OLLAMA_BASE_URL}/api/tags`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const models: ModelEntry[] = data.models || [];
    localModelNames = new Set(models.map((m) => m.name));
    return models;
  } catch (err) {
    console.log(`[ollama] local unavailable: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

async function fetchModelDetails(name: string): Promise<ModelEntry | null> {
  try {
    const res = await fetch(`${OLLAMA_BASE_URL}/api/show`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
  // 1. Explicit capability flag (Ollama >= 0.6)
  if (capabilities.some((c) => c === "vision")) return true;
  // 2. Known vision architecture prefixes
  const arch = String(modelInfo["general.architecture"] ?? "").toLowerCase();
  if (["llava", "bakllava", "moondream", "llava-next", "minicpm-v", "phi3-v", "mllama"].some((v) => arch.includes(v))) return true;
  // 3. CLIP vision encoder flag
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

  try {
    // already installed?
    const res = await fetch(`${OLLAMA_BASE_URL}/api/tags`);
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
      const pullRes = await fetch(`${OLLAMA_BASE_URL}/api/pull`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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

  // fetch details for all models (parallel, 5s timeout)
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
    baseUrl: `${OLLAMA_BASE_URL}/v1`,
    api: "openai-completions",
    apiKey: "ollama",
    compat: { supportsDeveloperRole: false, supportsReasoningEffort: false },
    models: modelConfigs,
  });

  writeModelCache(modelConfigs);
}

// ── entry ──

export default function (pi: ExtensionAPI) {
  // Register from cache synchronously (instant), then refresh from Ollama.
  // If cache is empty, the refresh runs immediately so models are available
  // before the session needs them.
  const hasCache = registerFromCache(pi);

  const ready = registerOllamaProvider(pi);
  if (!hasCache) {
    // No cache — we must block here so models are available before session starts
    // (extensions can return a promise to delay session init)
    return ready;
  }

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
}