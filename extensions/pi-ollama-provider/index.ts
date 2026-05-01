/**
 * Ollama Auto-Discovery Extension for Pi
 *
 * Discovers models from local Ollama or ollama.com cloud and registers
 * them via pi.registerProvider(). Uses native /api/chat streaming
 * (Option C hybrid) with fallback to OpenAI compat for old Ollama versions.
 *
 * Key improvements over simple openai-compat approach:
 * - Native /api/chat streaming fixes tool-calling bug (ollama#12557)
 * - Always sets num_ctx from model data (avoids 4096 silent truncation)
 * - Ghost-token retry for broken streaming responses
 * - Ollama context overflow detection
 * - Separate ollama and ollama-cloud providers
 * - Two-tier model discovery (cache + API)
 * - Tool capability detection
 * - /ollama-status, /ollama-info, /ollama-pull commands
 * - Sampling parameter settings (temperature, top_p, etc.)
 * - OLLAMA_HOST env var support
 *
 * Auth storage: uses pi's shared ~/.pi/agent/auth.json (not separate config).
 * In command handlers we go through ctx.modelRegistry.authStorage for
 * file-lock safety; at startup the factory reads auth.json directly.
 */

import type {
  ExtensionAPI,
  AssistantMessageEventStream,
} from "@mariozechner/pi-coding-agent";

import { resolveConfig, readOllamaAuthFromJson, type OllamaConfig, DEFAULT_LOCAL_URL, DEFAULT_CLOUD_URL } from "./auth.js";
import { getOllamaHost, discoverModels, readModelCache, writeModelCache, type OllamaModelConfig } from "./discovery.js";
import { registerCloudProvider, registerCloudTools } from "./cloud.js";
import { streamNativeChat, convertMessages, convertTools, parseNDJSON, isGhostTokenStream, isOllamaContextOverflow, type OllamaOptions, type OllamaChatChunk } from "./native-stream.js";
import { isOllamaContextOverflow as isOverflowFromSafety, calculateNumCtx, getDefaultKeepAlive } from "./context-safety.js";
import { readSettings, writeSettings, handleStatusCommand, handleInfoCommand, handlePullCommand, runSetupWizard, type OllamaSettings } from "./commands.js";

// Re-export for testing
export { resolveConfig, readOllamaAuthFromJson, DEFAULT_LOCAL_URL, DEFAULT_CLOUD_URL } from "./auth.js";
export type { OllamaConfig } from "./auth.js";
export { hasVision, hasToolSupport, hasReasoning, isCloudModel, generateModelId, extractContextLength, getOllamaHost } from "./discovery.js";
export { parseNDJSON, convertMessages, convertTools, isGhostTokenStream, isOllamaContextOverflow } from "./native-stream.js";
export { isOllamaContextOverflow as isOllamaOverflowFromSafety, calculateNumCtx, getDefaultKeepAlive } from "./context-safety.js";
export { readSettings, writeSettings, type OllamaSettings } from "./commands.js";

const LOCAL_PROVIDER_NAME = "ollama";
const CLOUD_PROVIDER_NAME = "ollama-cloud";

let currentConfig: OllamaConfig = resolveConfig();
let localModelNames = new Set<string>();
let lastLoadedModels: string[] = [];

// ── native streamSimple function ──

/**
 * Create a streamSimple function for pi.registerProvider that uses
 * Ollama's native /api/chat endpoint instead of the OpenAI compat shim.
 *
 * This fixes the tool-calling bug (ollama#12557) and enables
 * Ollama-specific options like num_ctx, temperature, top_p, etc.
 */
function createNativeStreamSimple(
  baseUrl: string,
  apiKey: string,
  settings: OllamaSettings,
) {
  return (
    model: { id: string; contextWindow?: number; input?: string[] },
    context: {
      messages: Array<Record<string, unknown>>;
      tools?: Array<Record<string, unknown>>;
    },
    stream: AssistantMessageEventStream,
    options?: { signal?: AbortSignal },
  ) => {
    const modelSupportsVision = Array.isArray(model.input) && model.input.includes("image");
    const contextWindow = model.contextWindow || 32768;

    // Build Ollama-specific options from settings
    const ollamaOptions: OllamaOptions = {
      num_ctx: settings.defaultNumCtx ?? calculateNumCtx(contextWindow, undefined),
      ...(settings.options || {}),
    };

    streamNativeChat(stream, {
      baseUrl,
      apiKey,
      model: model.id,
      contextWindow: ollamaOptions.num_ctx ?? contextWindow,
      messages: context.messages,
      tools: context.tools,
      modelSupportsVision,
      ollamaOptions,
      signal: options?.signal,
      keepAlive: settings.keepAlive || getDefaultKeepAlive(),
    });
  };
}

// ── provider registration ──

/**
 * Register the local Ollama provider with pi.
 * Uses native /api/chat streaming for tool-calling correctness,
 * with fallback to openai-compat if native fails.
 */
async function registerLocalProvider(pi: ExtensionAPI, settings: OllamaSettings): Promise<void> {
  const baseUrl = getOllamaHost();
  const config = currentConfig;

  // Two-tier discovery: cache-first, then API refresh
  const result = await discoverModels(
    baseUrl,
    config.apiKey,
    config.mode,
    localModelNames,
  );

  if (result.localModels.length === 0) {
    // No models discovered — leave cache-registered models if any
    return;
  }

  localModelNames = result.localModelNames;
  lastLoadedModels = result.loadedModels;

  // Filter: only register models with toolSupport if the user has that preference
  // (by default, register all models — user can filter in pi's model picker)
  const modelConfigs = result.localModels;

  // Build pi-compatible model list
  const piModels = modelConfigs.map((m) => ({
    id: m.id,
    name: m.name,
    reasoning: m.reasoning,
    input: m.input,
    contextWindow: m.contextWindow,
    maxTokens: m.maxTokens,
    cost: m.cost,
  }));

  // Use native streaming or openai-compat based on settings
  if (settings.streamingMode === "native") {
    pi.unregisterProvider(LOCAL_PROVIDER_NAME);
    pi.registerProvider(LOCAL_PROVIDER_NAME, {
      baseUrl: `${config.mode === "cloud" ? config.baseUrl : baseUrl}/v1`,
      api: "openai-completions",
      apiKey: config.apiKey,
      compat: {
        supportsDeveloperRole: false,
        supportsReasoningEffort: false,
      },
      models: piModels,
      streamSimple: createNativeStreamSimple(
        config.mode === "cloud" ? config.baseUrl : baseUrl,
        config.apiKey,
        settings,
      ),
    });
  } else {
    // OpenAI compat mode (fallback for old Ollama or user preference)
    pi.unregisterProvider(LOCAL_PROVIDER_NAME);
    pi.registerProvider(LOCAL_PROVIDER_NAME, {
      baseUrl: `${config.mode === "cloud" ? config.baseUrl : baseUrl}/v1`,
      api: "openai-completions",
      apiKey: config.apiKey,
      compat: {
        supportsDeveloperRole: false,
        supportsReasoningEffort: false,
      },
      models: piModels,
    });
  }

  // Update cache for instant next startup
  writeModelCache(modelConfigs);

  console.log(
    `[ollama] ${modelConfigs.length} models registered (streaming: ${settings.streamingMode || "native"})`,
  );
}

/**
 * Register cached models immediately for instant startup.
 * Returns true if cache was used (no blocking).
 */
function registerFromCache(pi: ExtensionAPI): boolean {
  const cached = readModelCache();
  if (!cached || cached.length === 0) return false;

  const settings = readSettings();
  const piModels = cached.map((m) => ({
    id: m.id,
    name: m.name,
    reasoning: m.reasoning,
    input: m.input,
    contextWindow: m.contextWindow,
    maxTokens: m.maxTokens,
    cost: m.cost,
  }));

  pi.unregisterProvider(LOCAL_PROVIDER_NAME);

  if (settings.streamingMode === "native") {
    pi.registerProvider(LOCAL_PROVIDER_NAME, {
      baseUrl: `${currentConfig.baseUrl}/v1`,
      api: "openai-completions",
      apiKey: currentConfig.apiKey,
      compat: {
        supportsDeveloperRole: false,
        supportsReasoningEffort: false,
      },
      models: piModels,
      streamSimple: createNativeStreamSimple(
        currentConfig.mode === "cloud"
          ? currentConfig.baseUrl
          : getOllamaHost(),
        currentConfig.apiKey,
        settings,
      ),
    });
  } else {
    pi.registerProvider(LOCAL_PROVIDER_NAME, {
      baseUrl: `${currentConfig.baseUrl}/v1`,
      api: "openai-completions",
      apiKey: currentConfig.apiKey,
      compat: {
        supportsDeveloperRole: false,
        supportsReasoningEffort: false,
      },
      models: piModels,
    });
  }

  console.log(`[ollama] ${cached.length} models from cache`);
  return true;
}

// ── model_select handler ──

function handleModelSelect(
  pi: ExtensionAPI,
  settings: OllamaSettings,
): (event: any) => Promise<void> {
  return async (event: any) => {
    if (event.model?.provider !== LOCAL_PROVIDER_NAME) return;
    const modelId = event.model?.id;
    if (!modelId || localModelNames.has(modelId)) return;

    // Auto-pull disabled?
    if (settings.autoPull === false) {
      console.log(`[ollama] auto-pull disabled; ${modelId} not available locally`);
      return;
    }

    // Interactive confirmation for large downloads
    try {
      await handlePullCommand(modelId, event, getOllamaHost(), currentConfig.apiKey);
    } catch (err) {
      console.error(`[ollama] auto-pull failed: ${modelId}: ${err}`);
    }
  };
}

// ── main entry ──

export default function (pi: ExtensionAPI) {
  // Resolve config from auth.json + env vars
  currentConfig = resolveConfig();

  if (currentConfig.apiKey !== "ollama") {
    console.log(
      `[ollama] API key from ${readOllamaAuthFromJson() ? "auth.json" : "OLLAMA_API_KEY env"}`,
    );
  }

  // Read extension settings
  const settings = readSettings();

  // ── Register local Ollama provider ──

  // Cache-first: register from cache immediately (no blocking)
  const hasCache = registerFromCache(pi);

  // Background refresh from API (non-blocking if cache hit)
  const ready = registerLocalProvider(pi, settings);

  // If no cache, block on discovery (models must be ready before session starts)
  if (!hasCache) {
    return ready;
  }

  // ── Register cloud provider (if API key is configured) ──

  if (currentConfig.mode === "cloud" || currentConfig.apiKey !== "ollama") {
    registerCloudProvider(pi, currentConfig.apiKey).then((count) => {
      if (count > 0) {
        registerCloudTools(pi, currentConfig.apiKey);
      }
    });
  }

  // ── Register commands ──

  pi.registerCommand("ollama-setup", {
    description: "Interactive setup wizard for Ollama (local or cloud)",
    handler: async (_args, ctx) => {
      await runSetupWizard(pi, ctx, {
        localBaseUrl: getOllamaHost(),
        cloudBaseUrl: DEFAULT_CLOUD_URL,
        apiKey: currentConfig.apiKey,
        authStorage: ctx.modelRegistry.authStorage,
      }, async (mode, baseUrl, apiKey) => {
        currentConfig = { mode, baseUrl, apiKey };
        await registerLocalProvider(pi, readSettings());
      });
    },
  });

  pi.registerCommand("ollama-refresh", {
    description: "Re-discover Ollama models",
    handler: async (_args, ctx) => {
      ctx.ui.notify("[ollama] Discovering models...", "info");
      await registerLocalProvider(pi, readSettings());
    },
  });

  pi.registerCommand("ollama-pull", {
    description: "Pull an Ollama model with progress bar",
    handler: async (args, ctx) => {
      const name =
        typeof args === "string" ? args.trim() : (args as any)?.model || (args as any)?.[0];
      if (!name) {
        ctx.ui.notify("Usage: /ollama-pull <model-name>", "error");
        return;
      }
      try {
        await handlePullCommand(name, ctx, getOllamaHost(), currentConfig.apiKey);
        // Refresh model list after pull
        await registerLocalProvider(pi, readSettings());
      } catch (err) {
        ctx.ui.notify(
          `Pull failed: ${err instanceof Error ? err.message : String(err)}`,
          "error",
        );
      }
    },
  });

  pi.registerCommand("ollama-status", {
    description: "Check Ollama connection status and loaded models",
    handler: async (_args, ctx) => {
      await handleStatusCommand(
        pi,
        ctx,
        getOllamaHost(),
        currentConfig.apiKey,
        localModelNames,
        lastLoadedModels,
      );
    },
  });

  pi.registerCommand("ollama-info", {
    description: "Show detailed model info from /api/show (e.g., /ollama-info llama3.1:8b)",
    handler: async (args, ctx) => {
      await handleInfoCommand(args, ctx, getOllamaHost(), currentConfig.apiKey);
    },
  });

  // ── Event handlers ──

  pi.on("model_select", handleModelSelect(pi, settings));
}