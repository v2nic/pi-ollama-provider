/**
 * Integration tests for the full extension module.
 * Tests the export/re-export surface and wiring between modules.
 */

import { describe, it, expect } from "vitest";

// Verify all public exports from index.ts
import {
  // auth re-exports
  resolveConfig,
  readOllamaAuthFromJson,
  DEFAULT_LOCAL_URL,
  DEFAULT_CLOUD_URL,
  // discovery re-exports
  hasVision,
  hasToolSupport,
  hasReasoning,
  isCloudModel,
  generateModelId,
  extractContextLength,
  getOllamaHost,
  // native-stream re-exports
  parseNDJSON,
  convertMessages,
  convertTools,
  isGhostTokenStream,
  isOllamaContextOverflow,
  // context-safety re-exports
  isOllamaOverflowFromSafety,
  calculateNumCtx,
  getDefaultKeepAlive,
  // commands re-exports
  readSettings,
  writeSettings,
} from "../extensions/pi-ollama-provider/index.js";

describe("Extension module exports", () => {
  it("re-exports auth functions", () => {
    expect(typeof resolveConfig).toBe("function");
    expect(typeof readOllamaAuthFromJson).toBe("function");
    expect(DEFAULT_LOCAL_URL).toBe("http://localhost:11434");
    expect(DEFAULT_CLOUD_URL).toBe("https://ollama.com");
  });

  it("re-exports discovery functions", () => {
    expect(typeof hasVision).toBe("function");
    expect(typeof hasToolSupport).toBe("function");
    expect(typeof hasReasoning).toBe("function");
    expect(typeof isCloudModel).toBe("function");
    expect(typeof generateModelId).toBe("function");
    expect(typeof extractContextLength).toBe("function");
    expect(typeof getOllamaHost).toBe("function");
  });

  it("re-exports native-stream functions", () => {
    expect(typeof parseNDJSON).toBe("function");
    expect(typeof convertMessages).toBe("function");
    expect(typeof convertTools).toBe("function");
    expect(typeof isGhostTokenStream).toBe("function");
    expect(typeof isOllamaContextOverflow).toBe("function");
  });

  it("re-exports context-safety functions", () => {
    expect(typeof isOllamaOverflowFromSafety).toBe("function");
    expect(typeof calculateNumCtx).toBe("function");
    expect(typeof getDefaultKeepAlive).toBe("function");
  });

  it("re-exports command functions", () => {
    expect(typeof readSettings).toBe("function");
    expect(typeof writeSettings).toBe("function");
  });
});

describe("Cross-module compatibility", () => {
  it("both overflow detection functions agree on positive case", () => {
    expect(isOllamaContextOverflow("exceeded max context length by 500 tokens")).toBe(true);
    expect(isOllamaOverflowFromSafety("exceeded max context length by 500 tokens")).toBe(true);
  });

  it("both overflow detection functions agree on negative case", () => {
    expect(isOllamaContextOverflow("model not found")).toBe(false);
    expect(isOllamaOverflowFromSafety("model not found")).toBe(false);
  });

  it("tool support capability inference with ollama#12557 context", () => {
    // Verify that models affected by the tool-calling bug are detected
    expect(hasToolSupport([], {}, "qwen2.5")).toBe(true);
    expect(hasToolSupport([], {}, "llama3.1")).toBe(true);
    expect(hasToolSupport([], {}, "gemma4")).toBe(true);
    // And models that don't support tools
    expect(hasToolSupport([], {}, "unknown-family")).toBe(false);
  });

  it("context safety: 4096 default is never used by calculateNumCtx", () => {
    // The whole point: we never use 4096 as default
    // calculateNumCtx returns 32768 for unknown models
    expect(calculateNumCtx(null)).toBe(32768);
    expect(calculateNumCtx(0)).toBe(32768);
    // And uses model's actual context for known models
    expect(calculateNumCtx(131072)).toBe(131072);
  });
});