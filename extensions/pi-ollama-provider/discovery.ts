/**
 * Ollama model discovery and capability inference.
 *
 * Two-tier discovery:
 *   1. Cache-first — read from ~/.pi/agent/ollama-models-cache.json (instant, no network)
 *   2. Fresh API — GET /api/tags + POST /api/show per model (background refresh)
 *
 * Capability inference:
 *   - Tool support: capabilities.includes("tools") or known tool-capable families
 *   - Vision: capabilities.includes("vision"), CLIP flag, or known vision architectures
 *   - Reasoning: capabilities.includes("thinking") or model name patterns
 *   - Context window: model_info.*.context_length from /api/show
 *
 * Also supports OLLAMA_HOST env var for custom server location.
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

// ── types ──

export interface OllamaModelConfig {
  id: string;
  name: string;
  reasoning: boolean;
  input: ("text" | "image")[];
  contextWindow: number;
  maxTokens: number;
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
  /** Whether this model supports tool/function calling */
  toolSupport: boolean;
  /** Whether this is a cloud model */
  isCloud: boolean;
  /** Raw Ollama model name (e.g., "llama3.1:8b") */
  ollamaName: string;
  /** Raw model details from /api/show for /ollama-info command */
  rawDetails?: {
    capabilities?: string[];
    modelInfo?: Record<string, unknown>;
    parameterSize?: string;
    quantizationLevel?: string;
    family?: string;
    families?: string[];
  };
}

export interface OllamaTagsModel {
  name: string;
  model: string;
  modified_at: string;
  size: number;
  digest: string;
  details: {
    parent_model?: string;
    format?: string;
    family?: string;
    families?: string[];
    parameter_size?: string;
    quantization_level?: string;
    capabilities?: string[];
  };
}

export interface OllamaShowResponse {
  license?: string;
  modelfile?: string;
  parameters?: string;
  template?: string;
  system?: string;
  details: {
    parent_model?: string;
    format?: string;
    family?: string;
    families?: string[];
    parameter_size?: string;
    quantization_level?: string;
    capabilities?: string[];
  };
  model_info?: Record<string, unknown>;
  capabilities?: string[];
}

// ── constants ──

const DEFAULT_CONTEXT_WINDOW = 32768;
const DEFAULT_MAX_TOKENS = 32768;
const CLOUD_DEFAULT_CONTEXT = 131072;
const CLOUD_DEFAULT_MAX_TOKENS = 131072;

const CONFIG_DIR = join(homedir(), ".pi", "agent");
export const CACHE_PATH = join(CONFIG_DIR, "ollama-models-cache.json");

/** Model families that reliably support tool calling */
const TOOL_CAPABLE_FAMILIES = new Set([
  "llama3.1",
  "llama3.2",
  "llama3.3",
  "qwen2",
  "qwen2.5",
  "qwen3",
  "qwen3.5",
  "mistral",
  "command-r",
  "gemma4",
  "gemma2",
  "mixtral",
  "granite",
  "phi4",
  "hermes",
  "firefunction",
  "deepseek-v4",
  "deepseek-v3",
]);

/** Model families that support vision */
const VISION_ARCHITECTURES = [
  "llava",
  "bakllava",
  "moondream",
  "llava-next",
  "minicpm-v",
  "phi3-v",
  "mllama",
  "llama3.2-vision",
  "gemma4", // gemma4 variants support vision
  "qwen2.5-vl",
  "qwen3-vl",
];

/** Model name patterns that indicate reasoning capability */
const REASONING_PATTERNS = [
  /\b(r1|reasoning|think|deepthink)\b/i,
  /\bdeepseek-r1\b/i,
  /\bgemma4\b/i, // gemma4 has thinking mode
  /\bqwen3(\.\d+)?\b/i, // qwen3+ has thinking
];

// ── OLLAMA_HOST resolution ──

export function getOllamaHost(): string {
  const host = process.env.OLLAMA_HOST;
  if (!host) return "http://localhost:11434";

  // OLLAMA_HOST may or may not include the scheme
  if (host.startsWith("http://") || host.startsWith("https://")) {
    return host.replace(/\/+$/, ""); // strip trailing slash
  }

  return `http://${host.replace(/\/+$/, "")}`;
}

// ── capability inference (exported for testing) ──

/**
 * Determine if a model supports vision from capabilities and model_info.
 */
export function hasVision(
  capabilities: string[],
  modelInfo: Record<string, unknown>,
  family?: string,
): boolean {
  // Explicit capability flag (Ollama ≥ 0.6)
  if (capabilities.includes("vision")) return true;

  // CLIP flag
  if (modelInfo["clip.has_vision_encoder"] === true) return true;

  // Known vision architecture from model info
  const arch = String(modelInfo["general.architecture"] ?? "").toLowerCase();
  if (VISION_ARCHITECTURES.some((v) => arch.includes(v))) return true;

  // Family-based detection
  if (family) {
    const lowerFamily = family.toLowerCase();
    if (VISION_ARCHITECTURES.some((v) => lowerFamily.includes(v))) return true;
  }

  return false;
}

/**
 * Determine if a model supports tool/function calling.
 */
export function hasToolSupport(
  capabilities: string[],
  modelInfo: Record<string, unknown>,
  family?: string,
  modelName?: string,
): boolean {
  // Explicit capability flag
  if (capabilities.includes("tools")) return true;

  // Family-based detection
  if (family) {
    const lowerFamily = family.toLowerCase();
    for (const capable of TOOL_CAPABLE_FAMILIES) {
      if (lowerFamily.includes(capable)) return true;
    }
  }

  // Model name patterns
  if (modelName) {
    const lower = modelName.toLowerCase();
    if (lower.includes("tools") || lower.includes("function")) return true;
    for (const capable of TOOL_CAPABLE_FAMILIES) {
      if (lower.includes(capable)) return true;
    }
  }

  return false;
}

/**
 * Determine if a model has reasoning/thinking capability.
 */
export function hasReasoning(
  capabilities: string[],
  modelName?: string,
): boolean {
  if (capabilities.includes("thinking")) return true;
  if (!modelName) return false;
  return REASONING_PATTERNS.some((p) => p.test(modelName));
}

/**
 * Extract context length from model_info.
 * Looks for keys ending in ".context_length".
 */
export function extractContextLength(
  modelInfo: Record<string, unknown>,
): number | null {
  const ctxKeys = Object.keys(modelInfo).filter((k) =>
    k.endsWith(".context_length"),
  );
  if (ctxKeys.length === 0) return null;

  // Use the first context_length found
  const val = Number(modelInfo[ctxKeys[0]]);
  return Number.isFinite(val) && val > 0 ? val : null;
}

/**
 * Determine if a model is a cloud model.
 */
export function isCloudModel(
  modelName: string,
  localModelNames: Set<string>,
  modelSize: number,
  configMode: "local" | "cloud",
): boolean {
  if (configMode === "cloud") return true;
  if (modelName.includes(":cloud") || modelName.endsWith("-cloud")) return true;
  // Large unpulled models (≥100GB) are likely cloud
  if (!localModelNames.has(modelName) && modelSize > 100_000_000_000) return true;
  return false;
}

/**
 * Generate the model ID, adding cloud suffix if needed.
 */
export function generateModelId(modelName: string, isCloud: boolean): string {
  if (isCloud && !modelName.endsWith("-cloud") && !modelName.includes(":cloud")) {
    return modelName.includes(":") ? `${modelName}-cloud` : `${modelName}:cloud`;
  }
  return modelName;
}

// ── cache ──

export function readModelCache(): OllamaModelConfig[] | null {
  try {
    if (!existsSync(CACHE_PATH)) return null;
    const data = readFileSync(CACHE_PATH, "utf-8");
    const parsed = JSON.parse(data);
    return Array.isArray(parsed) && parsed.length > 0 ? parsed : null;
  } catch {
    return null;
  }
}

export function writeModelCache(models: OllamaModelConfig[]): void {
  try {
    mkdirSync(CONFIG_DIR, { recursive: true });
    writeFileSync(CACHE_PATH, JSON.stringify(models, null, 2), "utf-8");
  } catch (err) {
    console.log(
      `[ollama] cache write failed: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

// ── API fetching ──

export function authHeaders(apiKey?: string): Record<string, string> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey && apiKey !== "ollama") {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }
  return headers;
}

/**
 * Fetch the list of local models from /api/tags.
 */
export async function fetchLocalModels(
  baseUrl: string,
  apiKey?: string,
): Promise<OllamaTagsModel[]> {
  try {
    const res = await fetch(`${baseUrl}/api/tags`, {
      headers: authHeaders(apiKey),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return data.models || [];
  } catch (err) {
    console.log(
      `[ollama] unavailable (${baseUrl}): ${err instanceof Error ? err.message : String(err)}`,
    );
    return [];
  }
}

/**
 * Fetch detailed model info from /api/show.
 */
export async function fetchModelDetails(
  name: string,
  baseUrl: string,
  apiKey?: string,
): Promise<OllamaShowResponse | null> {
  try {
    const res = await fetch(`${baseUrl}/api/show`, {
      method: "POST",
      headers: authHeaders(apiKey),
      body: JSON.stringify({ name }),
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

/**
 * Check if the Ollama server is running and reachable.
 */
export async function checkOllamaRunning(baseUrl: string): Promise<boolean> {
  try {
    const res = await fetch(`${baseUrl}/api/tags`, {
      headers: authHeaders(),
      signal: AbortSignal.timeout(5000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Get currently loaded models from /api/ps.
 */
export async function fetchLoadedModels(
  baseUrl: string,
  apiKey?: string,
): Promise<string[]> {
  try {
    const res = await fetch(`${baseUrl}/api/ps`, {
      headers: authHeaders(apiKey),
    });
    if (!res.ok) return [];
    const data = await res.json();
    return (data.models || []).map((m: any) => m.name as string);
  } catch {
    return [];
  }
}

// ── full discovery ──

export interface DiscoveryResult {
  localModels: OllamaModelConfig[];
  localModelNames: Set<string>;
  loadedModels: string[];
}

/**
 * Full model discovery: /api/tags → /api/show per model.
 * Returns model configs ready for pi.registerProvider(), along with
 * the sets of local model names and currently-loaded models.
 */
export async function discoverModels(
  baseUrl: string,
  apiKey: string,
  configMode: "local" | "cloud",
  localModelNames?: Set<string>,
  detailTimeoutMs: number = 5000,
): Promise<DiscoveryResult> {
  const tagsModels = await fetchLocalModels(baseUrl, apiKey);
  if (tagsModels.length === 0) {
    return { localModels: [], localModelNames: new Set(), loadedModels: [] };
  }

  const discoveredNames = new Set(tagsModels.map((m) => m.name));
  const names = localModelNames ?? discoveredNames;

  // Fetch /api/show for each model with a timeout
  const detailResults = new Map<string, OllamaShowResponse | null>();
  await Promise.race([
    Promise.all(
      tagsModels.map(async (m) => {
        detailResults.set(m.name, await fetchModelDetails(m.name, baseUrl, apiKey));
      }),
    ),
    new Promise((r) => setTimeout(r, detailTimeoutMs)),
  ]);

  // Check which models are currently loaded (optional, best-effort)
  const loadedModels = await fetchLoadedModels(baseUrl, apiKey);

  console.log(`[ollama] ${tagsModels.length} models discovered`);

  const modelConfigs: OllamaModelConfig[] = tagsModels.map((model) => {
    const details = detailResults.get(model.name);
    const capabilities: string[] =
      details?.capabilities ||
      details?.details?.capabilities ||
      model.details?.capabilities ||
      [];
    const modelInfo: Record<string, unknown> = {
      ...(details?.model_info || {}),
    };

    const family = details?.details?.family || model.details?.family;
    const families = details?.details?.families || model.details?.families;
    const parameterSize =
      details?.details?.parameter_size || model.details?.parameter_size;
    const quantizationLevel =
      details?.details?.quantization_level || model.details?.quantization_level;

    // Context window from /api/show
    let contextWindow = DEFAULT_CONTEXT_WINDOW;
    let maxTokens = DEFAULT_MAX_TOKENS;
    const ctxLength = extractContextLength(modelInfo);
    if (ctxLength !== null) {
      contextWindow = ctxLength;
      maxTokens = Math.min(Math.round(ctxLength * 0.25), 131072);
    }

    // Cloud model detection
    const isCloud = isCloudModel(model.name, names, model.size, configMode);
    if (isCloud && ctxLength === null) {
      contextWindow = CLOUD_DEFAULT_CONTEXT;
      maxTokens = CLOUD_DEFAULT_MAX_TOKENS;
    }

    const modelId = generateModelId(model.name, isCloud);

    // Capability inference
    const vision = hasVision(capabilities, modelInfo, family);
    const toolSupport = hasToolSupport(capabilities, modelInfo, family, model.name);
    const reasoning = hasReasoning(capabilities, model.name);

    return {
      id: modelId,
      name: isCloud ? `${model.name} (cloud)` : model.name,
      reasoning,
      input: vision ? (["text", "image"] as const) : (["text"] as const),
      contextWindow,
      maxTokens,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      toolSupport,
      isCloud,
      ollamaName: model.name,
      rawDetails: {
        capabilities,
        modelInfo,
        parameterSize,
        quantizationLevel,
        family,
        families,
      },
    };
  });

  return {
    localModels: modelConfigs,
    localModelNames: names,
    loadedModels,
  };
}