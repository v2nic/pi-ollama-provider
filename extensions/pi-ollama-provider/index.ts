/**
 * Ollama Auto-Discovery Extension for Pi
 *
 * Discovers models from local Ollama or ollama.com cloud and registers
 * them via pi.registerProvider(). Uses native /api/chat streaming
 * with a custom API name to avoid conflicts with other providers.
 *
 * IMPORTANT: We use api: "ollama-native" instead of "openai-completions" to avoid
 * overwriting the openai-completions API provider that openrouter and others use.
 * The apiProviderRegistry in pi is keyed by API string, so using a unique API name
 * prevents our streamSimple from overwriting other providers.
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
 * - OLLAMA_API_BASE env var for custom cloud endpoint
 * - AuthStorage for file-lock-safe auth resolution
 * - Cache versioning for forward-compatible model discovery
 * - Atomic settings writes to prevent corruption
 * - Settings validation with warning logs
 * - Structured log buffer for TUI status bar (no console.log noise)
 *
 * Auth storage: uses pi's AuthStorage for full resolution priority:
 *   1. Runtime overrides (CLI --api-key flag)
 *   2. auth.json "ollama-cloud" credential
 *   3. auth.json "ollama" credential (fallback)
 *   4. OLLAMA_API_KEY env var
 */

import type {
  ExtensionAPI,
} from "@mariozechner/pi-coding-agent";
import type {
  AssistantMessageEventStream,
  Context,
  Model,
  SimpleStreamOptions,
  Api,
} from "@mariozechner/pi-ai";
import { createAssistantMessageEventStream } from "@mariozechner/pi-ai";

import {
  resolveConfig,
  resolveConfigAsync,
  readOllamaAuthFromJson,
  describeAuthSource,
  type OllamaConfig,
  DEFAULT_LOCAL_URL,
  DEFAULT_CLOUD_URL,
} from "./auth.js";
import {
  getOllamaHost,
  discoverModels,
  readModelCache,
  writeModelCache,
  assembleModelsFromCache,
  type OllamaModelConfig,
  type OllamaModelCache,
} from "./discovery.js";
import { registerCloudProvider, registerCloudTools } from "./cloud.js";
import { streamNativeChat } from "./native-stream.js";
import { isOllamaContextOverflow as isOverflowFromSafety, calculateNumCtx, getDefaultKeepAlive } from "./context-safety.js";
import { readSettings, writeSettings, handleStatusCommand, handleInfoCommand, handlePullCommand, runSetupWizard, type OllamaSettings } from "./commands.js";
import { logWarn, logError, getLogEntries, formatStatusLine } from "./log-buffer.js";

// Re-export for testing
export { resolveConfig, resolveConfigAsync, readOllamaAuthFromJson, describeAuthSource, DEFAULT_LOCAL_URL, DEFAULT_CLOUD_URL } from "./auth.js";
export type { OllamaConfig } from "./auth.js";
export { hasVision, hasToolSupport, hasReasoning, isCloudModel, generateModelId, extractContextLength, getOllamaHost, assembleModelsFromCache, type OllamaModelCache } from "./discovery.js";
export { parseNDJSON, convertMessages, convertTools, isGhostTokenStream, isOllamaContextOverflow } from "./native-stream.js";
export { isOllamaContextOverflow as isOllamaOverflowFromSafety, calculateNumCtx, getDefaultKeepAlive } from "./context-safety.js";
export { readSettings, writeSettings, validateSettings, type SettingsValidationIssue, type OllamaSettings } from "./commands.js";
export { createRenderResult, createApiKeyResolver, formatSearchResults, formatFetchResult, fetchCloudModels, fetchCloudModelDetails, registerCloudProvider, registerCloudTools } from "./cloud.js";
export type { SearchResult, FetchResult, ApiKeyResolver } from "./cloud.js";
export { logWarn, logError, getLogEntries, formatStatusLine, type LogEntry } from "./log-buffer.js";

const LOCAL_PROVIDER_NAME = "ollama";
const CLOUD_PROVIDER_NAME = "ollama-cloud";

// Custom API name to avoid conflicting with openai-completions
// The apiProviderRegistry is keyed by API string, so we need a unique name
// that won't overwrite openrouter, azure, etc.
const OLLAMA_NATIVE_API = "ollama-native";

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
    model: Model<Api>,
    context: Context,
    options?: SimpleStreamOptions,
  ): AssistantMessageEventStream => {
    const stream = createAssistantMessageEventStream();

    const modelSupportsVision = Array.isArray(model.input) && model.input.includes("image");
    const contextWindow = model.contextWindow || 32768;

    // Build Ollama-specific options from settings
    const ollamaOptions: Record<string, unknown> = {
      num_ctx: settings.defaultNumCtx ?? calculateNumCtx(contextWindow, undefined),
      ...(settings.options || {}),
    };

    streamNativeChat(stream, {
      baseUrl,
      apiKey,
      model: model.id,
      contextWindow: ollamaOptions.num_ctx as number ?? contextWindow,
      messages: context.messages as Array<Record<string, unknown>>,
      tools: context.tools as Array<Record<string, unknown>> | undefined,
      modelSupportsVision,
      ollamaOptions,
      signal: options?.signal,
      keepAlive: settings.keepAlive || getDefaultKeepAlive(),
    }).catch(() => {
      // Errors are already pushed to the stream via error events
    });

    return stream;
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

  // Build pi-compatible model list with our custom API
  const piModels = modelConfigs.map((m) => ({
    id: m.id,
    name: m.name,
    reasoning: m.reasoning,
    input: m.input,
    contextWindow: m.contextWindow,
    maxTokens: m.maxTokens,
    cost: m.cost,
    tools: m.toolSupport,
  }));

  pi.unregisterProvider(LOCAL_PROVIDER_NAME);

  // Use native streaming or openai-compat based on settings
  // Native uses custom API ("ollama-native") to avoid overwriting other providers
  // that use "openai-completions" (openrouter, azure)
  if (settings.streamingMode !== "openai-compat") {
    // Native streaming mode - uses custom API + streamSimple
    pi.registerProvider(LOCAL_PROVIDER_NAME, {
      baseUrl: `${config.mode === "cloud" ? config.baseUrl : baseUrl}/v1`,
      api: OLLAMA_NATIVE_API,
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

  // Update cache for instant next startup (raw API data)
  writeModelCache(result.rawCacheData);
}

/**
 * Register cached models immediately for instant startup.
 * Returns true if cache was used (no blocking).
 */
function registerFromCache(pi: ExtensionAPI): boolean {
  const cached = readModelCache();
  if (!cached) return false;

  const settings = readSettings();
  // Re-assemble model configs from raw cached API data
  // This means if capability inference logic changes, we can
  // re-process the same cached data without re-fetching.
  const modelConfigs = assembleModelsFromCache(cached, currentConfig.mode);
  if (modelConfigs.length === 0) return false;

  const piModels = modelConfigs.map((m) => ({
    id: m.id,
    name: m.name,
    reasoning: m.reasoning,
    input: m.input,
    contextWindow: m.contextWindow,
    maxTokens: m.maxTokens,
    cost: m.cost,
    tools: m.toolSupport,
  }));

  pi.unregisterProvider(LOCAL_PROVIDER_NAME);

  // Use native streaming or openai-compat based on settings
  // Native uses custom API ("ollama-native") to avoid overwriting other providers
  if (settings.streamingMode !== "openai-compat") {
    // Native streaming mode - uses custom API + streamSimple
    pi.registerProvider(LOCAL_PROVIDER_NAME, {
      baseUrl: `${currentConfig.baseUrl}/v1`,
      api: OLLAMA_NATIVE_API,
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
    // OpenAI compat mode (fallback for old Ollama or user preference)
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
    if (settings.autoPull === false) return;

    // Interactive confirmation for large downloads
    try {
      await handlePullCommand(modelId, event, getOllamaHost(), currentConfig.apiKey);
    } catch {
      // Auto-pull failed — non-fatal, user can pull manually
    }
  };
}

// ── main entry ──

export default async function (pi: ExtensionAPI) {
  // ── Phase 1: Synchronous startup (cache-first) ──

  // Resolve config with sync fallback for immediate cache registration
  currentConfig = resolveConfig();

  // Auth source (for /ollama-status display, not logged to TUI)
  const authSource = describeAuthSource(currentConfig);

  // Read extension settings (warnings are buffered, not logged)
  const settings = readSettings();

  // ── Register local Ollama provider with cache ──

  // Cache-first: register from cache immediately (no blocking)
  const hasCache = registerFromCache(pi);

  // ── Phase 2: Async AuthStorage resolution ──

  // Available through ctx.modelRegistry.authStorage in command handlers.
  // For startup, we use sync resolveConfig() which covers auth.json + env vars.
  // AuthStorage provides additional runtime override support (--api-key flag).
  // When we have access to authStorage asynchronously, we can refresh the config.

  // ── Register local provider (background refresh from API) ──

  let localProviderReady: Promise<void>;
  if (!hasCache) {
    // No cache — try to register immediately from API
    localProviderReady = registerLocalProvider(pi, settings).catch(() => {
      // Discovery failed — no local models available, cache still empty.
      // Status bar will show the error after session_start.
      logError("local", "API discovery failed, no models available");
    });
  } else {
    // Has cache — refresh in background
    localProviderReady = registerLocalProvider(pi, settings);
  }

  // ── Register cloud provider (if API key is configured) ──

  if (currentConfig.mode === "cloud" || currentConfig.apiKey !== "ollama") {
    registerCloudProvider(pi, currentConfig.apiKey).then((count) => {
      if (count > 0) {
        registerCloudTools(pi, currentConfig.apiKey);
      }
    }).catch(() => {
      // Cloud registration failed — non-fatal, local models still work.
      // Status bar will show the error after session_start.
      logError("cloud", "registration failed (cloud may be unreachable)");
    });

    // Cloud-only mode without cache — API registration is attempted above
  }

  // ── Register commands ──

  pi.registerCommand("ollama-setup", {
    description: "Interactive setup wizard for Ollama (local or cloud)",
    handler: async (_args, ctx) => {
      await runSetupWizard(pi, ctx, {
        localBaseUrl: getOllamaHost(),
        cloudBaseUrl: currentConfig.mode === "cloud" ? currentConfig.baseUrl : DEFAULT_CLOUD_URL,
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
      ctx.ui.notify("Discovering models...", "info");
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
        authSource,
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

  // ── Show buffered errors/warnings in status bar on session start ──

  pi.on("session_start", (_event, ctx) => {
    const status = formatStatusLine();
    ctx.ui.setStatus("ollama", status ?? undefined);
  });

  // Wait for local provider to be ready if no cache was available
  await localProviderReady;
}