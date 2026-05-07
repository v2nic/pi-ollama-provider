/**
 * Ollama Cloud provider and tools.
 *
 * Connects Pi directly to ollama.com/v1 (no local server needed).
 * Filters to only tool-capable models (coding agent requires tools).
 * Includes ollama_web_search and ollama_web_fetch tools with rich TUI rendering.
 *
 * Improvements over v2.0 (issue #8):
 *   - renderResult for collapsible, theme-aware, visually truncated tool output
 *   - max_results parameter for web search (1-10, default 5)
 *   - Structured error handling with isError flag
 *   - details field for structured result data alongside human-readable content
 *   - Web tools enabled by default (opt-out via PI_OLLAMA_WEB_TOOLS=0)
 *   - API key retrieval via AuthStorage in tool execute context (file-lock safe)
 *   - TypeBox parameter schemas
 *
 * Authentication:
 *   - OLLAMA_API_KEY env var
 *   - auth.json "ollama-cloud" / "ollama" credential
 *   - Fallback: prompt user via /ollama-setup
 */

import type { ExtensionAPI, Theme } from "@mariozechner/pi-coding-agent";
import { keyHint, truncateToVisualLines } from "@mariozechner/pi-coding-agent";
import { Text, truncateToWidth } from "@mariozechner/pi-tui";
import { Type } from "@sinclair/typebox";
import { homedir } from "node:os";
import { join } from "node:path";

const CLOUD_PROVIDER_NAME = "ollama-cloud";

/** Cloud base URL, configurable via OLLAMA_API_BASE env var */
export const CLOUD_BASE_URL = (process.env.OLLAMA_API_BASE?.replace(/\/+$/, "") || "https://ollama.com");

/** Web tools enabled by default; disable via PI_OLLAMA_WEB_TOOLS=0 or false */
const WEB_TOOLS_DISABLED =
  process.env.PI_OLLAMA_WEB_TOOLS === "0" ||
  process.env.PI_OLLAMA_WEB_TOOLS === "false";

// ── TUI rendering ──

/** Number of visual lines to show in collapsed mode */
const PREVIEW_LINES = 8;

/**
 * Build a renderResult handler that shows a truncated preview when collapsed
 * and the full output when expanded. Follows the bash tool pattern from pi core.
 *
 * Handles:
 * - Collapsed: shows last N visual lines with "N earlier lines, expand to see all" hint
 * - Expanded: shows full output with theme-aware coloring
 * - Errors: always expanded with error styling
 */
export function createRenderResult() {
  return (
    result: { content: Array<{ type: string; text: string }>; isError?: boolean },
    options: { expanded: boolean; isPartial: boolean },
    theme: Theme,
    context: {
      invalidate: () => void;
      lastComponent: import("@mariozechner/pi-tui").Component | undefined;
      state: { cachedWidth?: number; cachedLines?: string[]; cachedSkipped?: number };
    },
  ) => {
    const state = context.state;
    const output = result.content
      .map((c) => c.text)
      .join("")
      .trim();
    const styledOutput = output
      .split("\n")
      .map((line: string) => theme.fg("toolOutput", line))
      .join("\n");

    if (options.expanded || result.isError) {
      const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
      text.setText(result.isError ? styledOutput : `\n${styledOutput}`);
      return text;
    }

    return {
      render: (width: number) => {
        if (state.cachedWidth !== width) {
          const preview = truncateToVisualLines(styledOutput, PREVIEW_LINES, width);
          state.cachedLines = preview.visualLines;
          state.cachedSkipped = preview.skippedCount;
          state.cachedWidth = width;
        }
        if (state.cachedSkipped && state.cachedSkipped > 0) {
          const hint =
            theme.fg("muted", `... (${state.cachedSkipped} earlier lines,`) +
            ` ${keyHint("app.tools.expand", "to expand")})`;
          return ["", truncateToWidth(hint, width, "..."), ...(state.cachedLines ?? [])];
        }
        return ["", ...(state.cachedLines ?? [])];
      },
      invalidate: () => {
        state.cachedWidth = undefined;
        state.cachedLines = undefined;
        state.cachedSkipped = undefined;
      },
    };
  };
}

// ── cloud model discovery ──

interface CloudModel {
  id: string;
  name?: string;
  capabilities?: string[];
  context_length?: number;
}

/**
 * Fetch cloud models from ollama.com/v1/models.
 * Only returns models that support tool calling.
 */
export async function fetchCloudModels(
  apiKey: string,
): Promise<CloudModel[]> {
  try {
    const res = await fetch(`${CLOUD_BASE_URL}/v1/models`, {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
    });
    if (!res.ok) {
      console.log(`[ollama-cloud] /v1/models failed: HTTP ${res.status}`);
      return [];
    }
    const data = await res.json();
    const models: CloudModel[] = (data.data || []).map((m: any) => ({
      id: m.id,
      name: m.id,
      capabilities: m.capabilities || [],
      context_length: m.context_length,
    }));

    // Filter: only tool-capable models (coding agent requires tools)
    return models.filter((m) =>
      m.capabilities?.includes("tools") || isToolCapableByName(m.id),
    );
  } catch (err) {
    console.log(
      `[ollama-cloud] unavailable: ${err instanceof Error ? err.message : String(err)}`,
    );
    return [];
  }
}

/**
 * Fetch detailed model info from ollama.com /api/show.
 */
export async function fetchCloudModelDetails(
  modelName: string,
  apiKey: string,
): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch(`${CLOUD_BASE_URL}/api/show`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ name: modelName }),
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

/**
 * Heuristic: some model names indicate tool capability even without
 * the explicit capability flag.
 */
function isToolCapableByName(modelId: string): boolean {
  const lower = modelId.toLowerCase();
  const toolFamilies = [
    "llama3.1",
    "llama3.3",
    "qwen2.5",
    "qwen3",
    "qwen3.5",
    "mistral",
    "gemma4",
    "deepseek-v4",
    "command-r",
    "granite",
    "phi4",
    "kimi",
  ];
  return toolFamilies.some((f) => lower.includes(f));
}

/**
 * Heuristic: some model names indicate vision capability even without
 * the explicit capability flag.
 */
function isVisionCapableByName(modelId: string): boolean {
  const lower = modelId.toLowerCase();
  const visionFamilies = [
    "llava",
    "moondream",
    "minicpm-v",
    "phi3-v",
    "mllama",
    "llama3.2-vision",
    "gemma4",
    "qwen-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "kimi",
    "gemini",
  ];
  return visionFamilies.some((f) => lower.includes(f));
}

// ── cloud provider registration ──

/**
 * Register the ollama-cloud provider with pi.
 * Only includes tool-capable models with zero cost (subscription, not per-token).
 */
export async function registerCloudProvider(
  pi: ExtensionAPI,
  apiKey: string,
): Promise<number> {
  const clouds = await fetchCloudModels(apiKey);

  if (clouds.length === 0) {
    console.log("[ollama-cloud] no tool-capable models found");
    pi.unregisterProvider(CLOUD_PROVIDER_NAME);
    return 0;
  }

  const modelConfigs = clouds.map((model) => {
    const contextWindow = model.context_length || 131072;
    const isCloud = true;

    return {
      id: model.id,
      name: `${model.name || model.id} (cloud)`,
      reasoning:
        model.capabilities?.includes("thinking") ||
        /\b(r1|deepseek-r1|gemma4|qwen3)\b/i.test(model.id),
      input: (model.capabilities?.includes("vision") || isVisionCapableByName(model.id)
        ? ["text", "image"]
        : ["text"]) as ("text" | "image")[],
      contextWindow,
      maxTokens: Math.min(Math.round(contextWindow * 0.25), 131072),
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      toolSupport: true,
      isCloud,
      ollamaName: model.id,
    };
  });

  pi.unregisterProvider(CLOUD_PROVIDER_NAME);
  pi.registerProvider(CLOUD_PROVIDER_NAME, {
    baseUrl: `${CLOUD_BASE_URL}/v1`,
    api: "openai-completions",
    apiKey,
    compat: { supportsDeveloperRole: false, supportsReasoningEffort: false },
    models: modelConfigs,
  });

  console.log(`[ollama-cloud] ${modelConfigs.length} tool-capable models registered`);
  return modelConfigs.length;
}

// ── API key resolution ──

export interface ApiKeyResolver {
  getApiKey(): Promise<string | undefined>;
}

/**
 * Create an API key resolver that checks AuthStorage first, then env var.
 * This is the recommended pattern (file-lock safe, follows pi convention).
 */
export function createApiKeyResolver(
  authStorage: { getApiKeyForProvider?: (provider: string) => string | undefined | Promise<string | undefined>; getApiKey?: (provider: string) => string | undefined | Promise<string | undefined> },
): ApiKeyResolver {
  return {
    async getApiKey() {
      // Try AuthStorage first (supports getApiKeyForProvider or getApiKey)
      try {
        if (typeof authStorage.getApiKeyForProvider === "function") {
          const key = await authStorage.getApiKeyForProvider("ollama-cloud");
          if (key) return key;
        }
        if (typeof authStorage.getApiKey === "function") {
          const key = await authStorage.getApiKey("ollama-cloud");
          if (key) return key;
          // Also try just "ollama" key
          const ollamaKey = await authStorage.getApiKey("ollama");
          if (ollamaKey && ollamaKey !== "ollama") return ollamaKey;
        }
      } catch {
        // AuthStorage not available, fall through
      }

      // Fallback to OLLAMA_API_KEY env var
      const envKey = process.env.OLLAMA_API_KEY;
      if (envKey) return envKey;

      return undefined;
    },
  };
}

function noApiKeyError() {
  return {
    content: [
      {
        type: "text" as const,
        text: "Error: No Ollama Cloud API key configured. Set OLLAMA_API_KEY or run /ollama-setup.",
      },
    ],
    isError: true,
  };
}

// ── result formatting ──

export interface SearchResult {
  title: string;
  url: string;
  content: string;
}

export interface FetchResult {
  title: string;
  content: string;
  links: string[];
}

export function formatSearchResults(results: SearchResult[]): string {
  if (results.length === 0) return "No results found.";
  return results
    .map((r, i) => `${i + 1}. ${r.title}\n   URL: ${r.url}\n   ${r.content}`)
    .join("\n\n");
}

export function formatFetchResult(data: FetchResult): string {
  const lines: string[] = [
    `Title: ${data.title}`,
    "",
    "Content:",
    data.content,
  ];

  if (data.links && data.links.length > 0) {
    lines.push("", `Links found: ${data.links.length}`);
    for (const l of data.links.slice(0, 10)) {
      lines.push(`  - ${l}`);
    }
    if (data.links.length > 10) {
      lines.push(`  ... and ${data.links.length - 10} more`);
    }
  }

  return lines.join("\n");
}

// ── web search/fetch tools ──

/**
 * Register ollama_web_search and ollama_web_fetch tools for Ollama Cloud.
 * These use Ollama Cloud's hosted web_search and web_fetch endpoints,
 * which are only available for cloud models.
 *
 * Tool names are prefixed with "ollama_" to avoid collision with
 * any other web search extensions the user may have installed.
 *
 * Web tools are enabled by default. Set PI_OLLAMA_WEB_TOOLS=0 to disable.
 */
export function registerCloudTools(pi: ExtensionAPI, apiKey: string): void {
  if (WEB_TOOLS_DISABLED) {
    console.log("[ollama-cloud] web tools disabled (PI_OLLAMA_WEB_TOOLS=0)");
    return;
  }

  const renderResult = createRenderResult();

  // Shared API key resolver — uses module-level apiKey as fallback
  const fallbackResolver: ApiKeyResolver = {
    async getApiKey() {
      // Module-level apiKey from registerCloudTools (from auth.json / env var)
      if (apiKey && apiKey !== "ollama") return apiKey;
      const envKey = process.env.OLLAMA_API_KEY;
      if (envKey) return envKey;
      return undefined;
    },
  };

  // ── ollama_web_search ──

  pi.registerTool({
    name: "ollama_web_search",
    label: "Ollama Web Search",
    description:
      "Search the web for real-time information using Ollama Cloud's web search API. " +
      "Returns relevant results with titles, URLs, and content snippets. " +
      "Requires an Ollama Cloud API key.",
    parameters: Type.Object({
      query: Type.String({ description: "The search query to execute" }),
      max_results: Type.Optional(
        Type.Integer({
          description: "Maximum number of search results to return (default: 5, max: 10)",
          default: 5,
          minimum: 1,
          maximum: 10,
        }),
      ),
    }),
    async execute(
      _toolCallId: string,
      params: { query: string; max_results?: number },
      signal: AbortSignal | undefined,
      _onUpdate: unknown,
      ctx: { modelRegistry?: { authStorage?: any } },
    ) {
      // Resolve API key: try AuthStorage from context first, then fallback
      let resolvedApiKey: string | undefined;

      if (ctx?.modelRegistry?.authStorage) {
        const resolver = createApiKeyResolver(ctx.modelRegistry.authStorage);
        resolvedApiKey = await resolver.getApiKey();
      }

      if (!resolvedApiKey) {
        resolvedApiKey = await fallbackResolver.getApiKey();
      }

      if (!resolvedApiKey) return noApiKeyError();

      const query = String(params.query ?? "");
      if (!query) {
        return {
          content: [{ type: "text" as const, text: "Error: empty query" }],
          isError: true,
        };
      }

      const maxResults = Math.min(Math.max(params.max_results ?? 5, 1), 10);

      try {
        const res = await fetch(`${CLOUD_BASE_URL}/api/web_search`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${resolvedApiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ query, max_results: maxResults }),
          signal,
        });

        if (!res.ok) {
          const errorText = await res.text().catch(() => "");
          return {
            content: [
              {
                type: "text" as const,
                text: `Search API error (status ${res.status}): ${errorText || res.statusText}`,
              },
            ],
            isError: true,
          };
        }

        const data = await res.json();
        const results: SearchResult[] = data.results || [];
        const formatted = formatSearchResults(results);

        return {
          content: [{ type: "text" as const, text: formatted }],
          details: { results },
        };
      } catch (err) {
        return {
          content: [
            {
              type: "text" as const,
              text: `Web search failed: ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
          isError: true,
        };
      }
    },
    renderResult,
  });

  // ── ollama_web_fetch ──

  pi.registerTool({
    name: "ollama_web_fetch",
    label: "Ollama Web Fetch",
    description:
      "Fetch and extract text content from a web page URL using Ollama Cloud's web fetch API. " +
      "Returns the page title, main content, and links found on the page. " +
      "Requires an Ollama Cloud API key.",
    parameters: Type.Object({
      url: Type.String({
        description: "URL to fetch and extract content from",
        format: "uri",
      }),
    }),
    async execute(
      _toolCallId: string,
      params: { url: string },
      signal: AbortSignal | undefined,
      _onUpdate: unknown,
      ctx: { modelRegistry?: { authStorage?: any } },
    ) {
      // Resolve API key: try AuthStorage from context first, then fallback
      let resolvedApiKey: string | undefined;

      if (ctx?.modelRegistry?.authStorage) {
        const resolver = createApiKeyResolver(ctx.modelRegistry.authStorage);
        resolvedApiKey = await resolver.getApiKey();
      }

      if (!resolvedApiKey) {
        resolvedApiKey = await fallbackResolver.getApiKey();
      }

      if (!resolvedApiKey) return noApiKeyError();

      const url = String(params.url ?? "");
      if (!url) {
        return {
          content: [{ type: "text" as const, text: "Error: empty URL" }],
          isError: true,
        };
      }

      try {
        const res = await fetch(`${CLOUD_BASE_URL}/api/web_fetch`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${resolvedApiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ url }),
          signal,
        });

        if (!res.ok) {
          const errorText = await res.text().catch(() => "");
          return {
            content: [
              {
                type: "text" as const,
                text: `Fetch API error (status ${res.status}): ${errorText || res.statusText}`,
              },
            ],
            isError: true,
          };
        }

        const data = await res.json();
        const fetchResult: FetchResult = {
          title: data.title || url,
          content: data.content || "",
          links: data.links || [],
        };
        const formatted = formatFetchResult(fetchResult);

        return {
          content: [{ type: "text" as const, text: formatted }],
          details: { title: fetchResult.title, content: fetchResult.content, links: fetchResult.links },
        };
      } catch (err) {
        return {
          content: [
            {
              type: "text" as const,
              text: `Web fetch failed: ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
          isError: true,
        };
      }
    },
    renderResult,
  });

  console.log("[ollama-cloud] ollama_web_search + ollama_web_fetch tools registered (enabled by default)");
}