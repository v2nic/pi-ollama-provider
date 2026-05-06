/**
 * Integration tests for the full extension module.
 * Tests the export/re-export surface and wiring between modules.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";

// Verify all public exports from index.ts
import {
  // auth re-exports
  resolveConfig,
  resolveConfigAsync,
  readOllamaAuthFromJson,
  describeAuthSource,
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
  assembleModelsFromCache,
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
  validateSettings,
} from "../extensions/pi-ollama-provider/index.js";

describe("Extension module exports", () => {
  it("re-exports auth functions", () => {
    expect(typeof resolveConfig).toBe("function");
    expect(typeof resolveConfigAsync).toBe("function");
    expect(typeof readOllamaAuthFromJson).toBe("function");
    expect(typeof describeAuthSource).toBe("function");
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
    expect(typeof assembleModelsFromCache).toBe("function");
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

  it("re-exports command functions including validateSettings", () => {
    expect(typeof readSettings).toBe("function");
    expect(typeof writeSettings).toBe("function");
    expect(typeof validateSettings).toBe("function");
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

describe("Auth new features integration", () => {
  const originalApiKey = process.env.OLLAMA_API_KEY;

  beforeEach(() => {
    delete process.env.OLLAMA_API_KEY;
  });

  afterEach(() => {
    if (originalApiKey !== undefined) {
      process.env.OLLAMA_API_KEY = originalApiKey;
    } else {
      delete process.env.OLLAMA_API_KEY;
    }
  });

  it("resolveConfigAsync returns default config for empty AuthStorage", async () => {
    const emptyStore = { getApiKey: async () => undefined };
    const config = await resolveConfigAsync(emptyStore);
    expect(config.mode).toBe("local");
    expect(config.apiKey).toBe("ollama");
    expect(config.authSource).toBe("default");
  });

  it("describeAuthSource returns readable labels", () => {
    expect(describeAuthSource({ mode: "local", baseUrl: DEFAULT_LOCAL_URL, apiKey: "ollama", authSource: "default" })).toBeTruthy();
  });

  it("validateSettings is callable and returns valid defaults", () => {
    const { validated, issues } = validateSettings({});
    expect(validated.streamingMode).toBe("native");
    expect(validated.version).toBe(1);
    expect(issues).toHaveLength(0);
  });
});