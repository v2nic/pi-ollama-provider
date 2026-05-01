/**
 * Ollama Cloud provider and tools.
 *
 * Connects Pi directly to ollama.com/v1 (no local server needed).
 * Filters to only tool-capable models (coding agent requires tools).
 * Includes web_search and web_fetch tools powered by Ollama Cloud API.
 *
 * Authentication:
 *   - OLLAMA_API_KEY env var
 *   - auth.json "ollama-cloud" credential
 *   - Fallback: prompt user via /ollama-setup
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { homedir } from "node:os";
import { join } from "node:path";

const CLOUD_PROVIDER_NAME = "ollama-cloud";
const CLOUD_BASE_URL = "https://ollama.com";

/** Web search/fetch tools disabled by default; enable via PI_OLLAMA_WEB_TOOLS=1 */
const WEB_TOOLS_ENABLED =
  process.env.PI_OLLAMA_WEB_TOOLS === "1" ||
  process.env.PI_OLLAMA_WEB_TOOLS === "true";

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
      input: (model.capabilities?.includes("vision")
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

// ── web search/fetch tools ──

/**
 * Register web_search and web_fetch tools for Ollama Cloud.
 * These use Ollama Cloud's hosted web_search and web_fetch endpoints,
 * which are only available for cloud models.
 *
 * Tool names are prefixed with "ollama_" to avoid collision with
 * any other web search extensions the user may have installed.
 */
export function registerCloudTools(pi: ExtensionAPI, apiKey: string): void {
  if (!WEB_TOOLS_ENABLED) return;

  pi.registerTool({
    name: "ollama_web_search",
    description:
      "Search the web using Ollama Cloud. Returns relevant search results.",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Search query",
        },
      },
      required: ["query"],
    },
    execute: async (args: Record<string, unknown>) => {
      const query = String(args.query ?? "");
      if (!query) return "Error: empty query";

      try {
        const res = await fetch(`${CLOUD_BASE_URL}/api/web_search`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${apiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ query }),
        });

        if (!res.ok) {
          return `Web search failed: HTTP ${res.status}`;
        }

        const data = await res.json();
        return JSON.stringify(data, null, 2);
      } catch (err) {
        return `Web search error: ${err instanceof Error ? err.message : String(err)}`;
      }
    },
  });

  pi.registerTool({
    name: "ollama_web_fetch",
    description:
      "Fetch and extract content from a web page URL using Ollama Cloud.",
    parameters: {
      type: "object",
      properties: {
        url: {
          type: "string",
          description: "URL to fetch",
        },
      },
      required: ["url"],
    },
    execute: async (args: Record<string, unknown>) => {
      const url = String(args.url ?? "");
      if (!url) return "Error: empty URL";

      try {
        const res = await fetch(`${CLOUD_BASE_URL}/api/web_fetch`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${apiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ url }),
        });

        if (!res.ok) {
          return `Web fetch failed: HTTP ${res.status}`;
        }

        const data = await res.json();
        return JSON.stringify(data, null, 2);
      } catch (err) {
        return `Web fetch error: ${err instanceof Error ? err.message : String(err)}`;
      }
    },
  });

  console.log("[ollama-cloud] web_search + web_fetch tools registered");
}