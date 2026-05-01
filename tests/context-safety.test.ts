/**
 * Tests for context-safety.ts — overflow detection, num_ctx calculation.
 */

import { describe, it, expect, afterEach } from "vitest";

import {
  isOllamaContextOverflow,
  extractOverflowTokens,
  calculateNumCtx,
  getDefaultKeepAlive,
} from "../extensions/pi-ollama-provider/context-safety.js";

// ════════════════════════════════════════════════════════════════
// isOllamaContextOverflow (from context-safety.ts, same logic as native-stream.ts)
// ════════════════════════════════════════════════════════════════

describe("isOllamaContextOverflow (context-safety)", () => {
  it("detects 'exceeded max context length'", () => {
    expect(isOllamaContextOverflow("prompt too long; exceeded max context length by 500 tokens")).toBe(true);
  });

  it("detects 'prompt too long'", () => {
    expect(isOllamaContextOverflow("prompt too long for model context")).toBe(true);
  });

  it("no match for unrelated errors", () => {
    expect(isOllamaContextOverflow("model not found")).toBe(false);
  });

  it("handles Error objects", () => {
    const err = new Error("exceeded max context length by 1000 tokens");
    expect(isOllamaContextOverflow(err)).toBe(true);
  });

  it("handles null/undefined", () => {
    expect(isOllamaContextOverflow(null)).toBe(false);
    expect(isOllamaContextOverflow(undefined)).toBe(false);
  });
});

// ════════════════════════════════════════════════════════════════
// extractOverflowTokens
// ════════════════════════════════════════════════════════════════

describe("extractOverflowTokens", () => {
  it("extracts token count from Ollama error format", () => {
    expect(extractOverflowTokens("prompt too long; exceeded max context length by 1234 tokens")).toBe(1234);
  });

  it("extracts token count with different wording", () => {
    expect(extractOverflowTokens("exceeded the context length by 500 tokens")).toBe(500);
  });

  it("returns null for no token count", () => {
    expect(extractOverflowTokens("context window exceeded")).toBeNull();
  });

  it("returns null for empty string", () => {
    expect(extractOverflowTokens("")).toBeNull();
  });
});

// ════════════════════════════════════════════════════════════════
// calculateNumCtx
// ════════════════════════════════════════════════════════════════

describe("calculateNumCtx", () => {
  it("returns model context length when available", () => {
    expect(calculateNumCtx(131072)).toBe(131072);
  });

  it("returns 32768 when context length is null", () => {
    expect(calculateNumCtx(null)).toBe(32768);
  });

  it("returns 32768 when context length is 0", () => {
    expect(calculateNumCtx(0)).toBe(32768);
  });

  it("returns 32768 when context length is negative", () => {
    expect(calculateNumCtx(-1)).toBe(32768);
  });

  it("caps at 131072 (128K) for safety", () => {
    expect(calculateNumCtx(999999)).toBe(131072);
  });

  it("does not cap for normal sizes", () => {
    expect(calculateNumCtx(8192)).toBe(8192);
    expect(calculateNumCtx(32768)).toBe(32768);
    expect(calculateNumCtx(65536)).toBe(65536);
  });
});

// ════════════════════════════════════════════════════════════════
// getDefaultKeepAlive
// ════════════════════════════════════════════════════════════════

describe("getDefaultKeepAlive", () => {
  const originalKeepAlive = process.env.OLLAMA_KEEP_ALIVE;

  afterEach(() => {
    if (originalKeepAlive !== undefined) {
      process.env.OLLAMA_KEEP_ALIVE = originalKeepAlive;
    } else {
      delete process.env.OLLAMA_KEEP_ALIVE;
    }
  });

  it("defaults to 30m for agent usage", () => {
    delete process.env.OLLAMA_KEEP_ALIVE;
    expect(getDefaultKeepAlive()).toBe("30m");
  });

  it("respects OLLAMA_KEEP_ALIVE env var", () => {
    process.env.OLLAMA_KEEP_ALIVE = "1h";
    expect(getDefaultKeepAlive()).toBe("1h");
  });
});