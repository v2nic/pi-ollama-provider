/**
 * Auth resolution for Ollama.
 *
 * Auth priority (per pi convention):
 *   1. CLI --api-key flag (handled by pi core, not here)
 *   2. auth.json "ollama" or "ollama-cloud" credential
 *   3. OLLAMA_API_KEY environment variable
 *   4. Default "ollama" (local, no auth)
 *
 * Uses pi's shared ~/.pi/agent/auth.json — not a separate config file.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

const AUTH_PATH = join(homedir(), ".pi", "agent", "auth.json");

export const DEFAULT_LOCAL_URL = "http://localhost:11434";
export const DEFAULT_CLOUD_URL = "https://ollama.com";

export interface OllamaConfig {
  mode: "local" | "cloud";
  baseUrl: string;
  apiKey: string;
}

/**
 * Read the "ollama" or "ollama-cloud" credential from a JSON auth file.
 * Accepts an explicit path for testing; defaults to the real auth.json.
 */
export function readOllamaAuthFromJson(
  authPath: string = AUTH_PATH,
): { type: "api_key"; key: string } | undefined {
  try {
    const data = readFileSync(authPath, "utf-8");
    const parsed = JSON.parse(data);
    // Check both "ollama-cloud" (preferred for cloud) and "ollama"
    const cred = parsed?.["ollama-cloud"] || parsed?.ollama;
    if (cred?.type === "api_key" && typeof cred.key === "string") {
      return cred;
    }
  } catch {}
  return undefined;
}

/**
 * Resolve the active Ollama config.
 *
 * Priority for API key:
 *   1. stored credential in auth.json (type: "api_key")
 *   2. OLLAMA_API_KEY environment variable
 *   3. default "ollama" (works for local / unauthenticated)
 *
 * Accepts explicit overrides for testing; defaults to real FS + env.
 */
export function resolveConfig(options?: {
  authPath?: string;
  envKey?: string;
}): OllamaConfig {
  const stored = readOllamaAuthFromJson(options?.authPath);
  const envKey = options?.envKey ?? process.env.OLLAMA_API_KEY;

  const apiKey = stored?.key || envKey || "ollama";

  if (stored) {
    const mode: "local" | "cloud" = apiKey !== "ollama" ? "cloud" : "local";
    const baseUrl = mode === "cloud" ? DEFAULT_CLOUD_URL : DEFAULT_LOCAL_URL;
    return { mode, baseUrl, apiKey };
  }

  if (envKey) {
    return { mode: "cloud", baseUrl: DEFAULT_CLOUD_URL, apiKey: envKey };
  }

  return { mode: "local", baseUrl: DEFAULT_LOCAL_URL, apiKey: "ollama" };
}

/**
 * Return auth headers for Ollama API requests.
 */
export function authHeaders(config?: OllamaConfig): Record<string, string> {
  const resolvedConfig = config ?? resolveConfig();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (resolvedConfig.apiKey && resolvedConfig.apiKey !== "ollama") {
    headers["Authorization"] = `Bearer ${resolvedConfig.apiKey}`;
  }
  return headers;
}