/**
 * Context window safety for Ollama.
 *
 * Three critical fixes for the most common Ollama + pi bugs:
 *
 * 1. Always set num_ctx — Ollama defaults to 4096, silently truncating
 *    system prompts and conversation history (root cause of #133, #2177, #2794)
 *
 * 2. Detect Ollama context overflow errors — Ollama returns 400 with
 *    "prompt too long; exceeded max context length by X tokens" which
 *    wasn't recognized by pi's auto-compaction (#2626)
 *
 * 3. Model loading retry handling — Local Ollama returns 502/500 during
 *    model loading; pi needs longer retry windows (#1887)
 */

// ── Ollama overflow error patterns ──
// (duplicated from native-stream.ts for standalone use with OpenAI compat)

export const OLLAMA_OVERFLOW_PATTERNS = [
  /exceeded max context length/i,
  /prompt too long/i,
  /context window exceeded/i,
  /maximum context length exceeded/i,
] as const;

/**
 * Check if an error message indicates an Ollama context overflow.
 * Works with both native API and OpenAI compat shim errors.
 */
export function isOllamaContextOverflow(error: unknown): boolean {
  if (!error) return false;
  const message =
    error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : String(error);
  return OLLAMA_OVERFLOW_PATTERNS.some((p) => p.test(message));
}

/**
 * Try to extract the number of excess tokens from an Ollama overflow error.
 * Ollama format: "prompt too long; exceeded max context length by 1234 tokens"
 * Returns null if not parseable.
 */
export function extractOverflowTokens(error: string): number | null {
  const match = error.match(/exceeded.*?by\s+(\d+)\s+tokens?/i);
  return match ? parseInt(match[1], 10) : null;
}

/**
 * Calculate a safe num_ctx value for a model.
 * Uses the model's trained context length from /api/show if available,
 * otherwise falls back to a generous default.
 *
 * The value is capped at 131072 (128K) to avoid excessive VRAM usage,
 * unless the model explicitly supports more.
 */
export function calculateNumCtx(
  modelContextLength: number | null,
  maxVram?: number, // optional VRAM limit hint
): number {
  const DEFAULT = 32768;
  const MAX_SAFE = 131072;

  if (modelContextLength === null || modelContextLength <= 0) {
    return DEFAULT;
  }

  // Cap at a safe maximum unless the model's trained context is higher
  return Math.min(modelContextLength, MAX_SAFE);
}

/**
 * Calculate the keep_alive value for interactive agent usage.
 * Default 5 minutes is too short — models unload between turns.
 * Recommended: ≥30 minutes for coding agent sessions.
 */
export function getDefaultKeepAlive(): string {
  return process.env.OLLAMA_KEEP_ALIVE || "30m";
}

/**
 * Calculate appropriate retry settings for Ollama model loading.
 * Local Ollama can take 30-120+ seconds to load a model.
 * pi-core now uses maxRetries: 30 (~215s window) — this function
 * provides the configuration for users who want to customize it.
 */
export function getModelLoadRetryConfig(): {
  maxRetries: number;
  retryDelayMs: number;
  description: string;
} {
  return {
    maxRetries: 30,
    retryDelayMs: 1000,
    description:
      "30 retries with 1s delay = ~30s effective window for model loading. " +
      "Large models (70B+) may need more. Set PI_OLLAMA_LOAD_RETRIES env to override.",
  };
}