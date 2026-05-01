/**
 * Tests for discovery.ts — model discovery, capability inference, cache.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { writeFileSync, mkdirSync, rmSync, existsSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  hasVision,
  hasToolSupport,
  hasReasoning,
  extractContextLength,
  isCloudModel,
  generateModelId,
  getOllamaHost,
  parseNDJSON,
} from "../extensions/pi-ollama-provider/index.js";

// ════════════════════════════════════════════════════════════════
// hasVision
// ════════════════════════════════════════════════════════════════

describe("hasVision", () => {
  it("detects from capabilities array", () => {
    expect(hasVision(["vision"], {})).toBe(true);
    expect(hasVision(["thinking"], {})).toBe(false);
    expect(hasVision(["vision", "thinking"], {})).toBe(true);
  });

  it("detects from known architecture", () => {
    expect(hasVision([], { "general.architecture": "llava-v1.6-34b" })).toBe(true);
    expect(hasVision([], { "general.architecture": "minicpm-v" })).toBe(true);
    expect(hasVision([], { "general.architecture": "mllama" })).toBe(true);
    expect(hasVision([], { "general.architecture": "llama" })).toBe(false);
  });

  it("detects from clip.has_vision_encoder", () => {
    expect(hasVision([], { "clip.has_vision_encoder": true })).toBe(true);
    expect(hasVision([], { "clip.has_vision_encoder": false })).toBe(false);
  });

  it("detects from family", () => {
    expect(hasVision([], {}, "llava")).toBe(true);
    expect(hasVision([], {}, "mllama")).toBe(true);
    expect(hasVision([], {}, "llama3.1")).toBe(false);
  });

  it("capabilities take precedence over family", () => {
    expect(hasVision(["vision"], {}, "llama3.1")).toBe(true);
  });
});

// ════════════════════════════════════════════════════════════════
// hasToolSupport
// ════════════════════════════════════════════════════════════════

describe("hasToolSupport", () => {
  it("detects from capabilities array", () => {
    expect(hasToolSupport(["tools"], {})).toBe(true);
    expect(hasToolSupport(["vision"], {})).toBe(false);
    expect(hasToolSupport(["tools", "vision"], {})).toBe(true);
  });

  it("detects from known families", () => {
    expect(hasToolSupport([], {}, "llama3.1")).toBe(true);
    expect(hasToolSupport([], {}, "qwen2.5")).toBe(true);
    expect(hasToolSupport([], {}, "gemma4")).toBe(true);
    expect(hasToolSupport([], {}, "mistral")).toBe(true);
    expect(hasToolSupport([], {}, "llama3")).toBe(false); // base llama3 is NOT tool-capable
  });

  it("detects from model name", () => {
    expect(hasToolSupport([], {}, undefined, "llama3.1:8b")).toBe(true);
    expect(hasToolSupport([], {}, undefined, "qwen2.5-coder:7b")).toBe(true);
    expect(hasToolSupport([], {}, undefined, "simple-lora:latest")).toBe(false);
  });

  it("capabilities take precedence", () => {
    expect(hasToolSupport(["tools"], {}, "unknown-family")).toBe(true);
  });
});

// ════════════════════════════════════════════════════════════════
// hasReasoning
// ════════════════════════════════════════════════════════════════

describe("hasReasoning", () => {
  it("detects from capabilities array", () => {
    expect(hasReasoning(["thinking"])).toBe(true);
    expect(hasReasoning(["vision"])).toBe(false);
  });

  it("detects from model name patterns", () => {
    expect(hasReasoning([], "deepseek-r1:671b")).toBe(true);
    expect(hasReasoning([], "qwen3:32b")).toBe(true);
    expect(hasReasoning([], "gemma4:27b")).toBe(true);
    expect(hasReasoning([], "llama3.1:8b")).toBe(false);
  });

  it("capabilities override name heuristics", () => {
    expect(hasReasoning(["thinking"], "llama3.1:8b")).toBe(true);
  });
});

// ════════════════════════════════════════════════════════════════
// extractContextLength
// ════════════════════════════════════════════════════════════════

describe("extractContextLength", () => {
  it("extracts from llama.context_length", () => {
    expect(extractContextLength({ "llama.context_length": 131072 })).toBe(131072);
  });

  it("extracts from qwen2.context_length", () => {
    expect(extractContextLength({ "qwen2.context_length": 32768 })).toBe(32768);
  });

  it("returns null when no context_length key", () => {
    expect(extractContextLength({ "llama.attention_head_count": 32 })).toBeNull();
  });

  it("returns null for invalid values", () => {
    expect(extractContextLength({ "llama.context_length": 0 })).toBeNull();
    expect(extractContextLength({ "llama.context_length": -1 })).toBeNull();
    expect(extractContextLength({ "llama.context_length": "big" })).toBeNull();
    expect(extractContextLength({ "llama.context_length": NaN })).toBeNull();
  });

  it("uses first context_length key when multiple exist", () => {
    const result = extractContextLength({
      "llama.context_length": 8192,
      "qwen.context_length": 32768,
    });
    // Should return one of them (first found)
    expect([8192, 32768]).toContain(result);
  });
});

// ════════════════════════════════════════════════════════════════
// isCloudModel
// ════════════════════════════════════════════════════════════════

describe("isCloudModel", () => {
  it("all models are cloud when mode=cloud", () => {
    expect(isCloudModel("llama3:8b", new Set(["llama3:8b"]), 4e9, "cloud")).toBe(true);
  });

  it(":cloud tag detected", () => {
    expect(isCloudModel("kimi-k2.6:cloud", new Set(["kimi-k2.6:cloud"]), 384, "local")).toBe(true);
  });

  it("-cloud suffix detected", () => {
    expect(isCloudModel("qwen3.5:397b-cloud", new Set(), 393, "local")).toBe(true);
  });

  it("local pulled models not flagged", () => {
    expect(isCloudModel("llama3:8b", new Set(["llama3:8b"]), 4.7e9, "local")).toBe(false);
  });

  it("large unpulled models flagged (size fallback)", () => {
    expect(isCloudModel("huge:200b", new Set(["llama3:8b"]), 200e9, "local")).toBe(true);
  });

  it("small unpulled local models not flagged", () => {
    expect(isCloudModel("tiny:1b", new Set(), 1e9, "local")).toBe(false);
  });
});

// ════════════════════════════════════════════════════════════════
// generateModelId
// ════════════════════════════════════════════════════════════════

describe("generateModelId", () => {
  it("preserves existing :cloud suffix", () => {
    expect(generateModelId("kimi-k2.6:cloud", true)).toBe("kimi-k2.6:cloud");
  });

  it("preserves existing -cloud suffix", () => {
    expect(generateModelId("qwen3.5:397b-cloud", true)).toBe("qwen3.5:397b-cloud");
  });

  it("adds -cloud for tagged models in cloud mode", () => {
    expect(generateModelId("qwen3.5:397b", true)).toBe("qwen3.5:397b-cloud");
  });

  it("adds :cloud for bare-name models in cloud mode", () => {
    expect(generateModelId("gemini-3-flash-preview", true)).toBe("gemini-3-flash-preview:cloud");
  });

  it("does not modify non-cloud models", () => {
    expect(generateModelId("llama3:8b", false)).toBe("llama3:8b");
  });
});

// ════════════════════════════════════════════════════════════════
// getOllamaHost
// ════════════════════════════════════════════════════════════════

describe("getOllamaHost", () => {
  const originalHost = process.env.OLLAMA_HOST;

  afterEach(() => {
    if (originalHost !== undefined) {
      process.env.OLLAMA_HOST = originalHost;
    } else {
      delete process.env.OLLAMA_HOST;
    }
  });

  it("defaults to localhost:11434 when OLLAMA_HOST not set", () => {
    delete process.env.OLLAMA_HOST;
    expect(getOllamaHost()).toBe("http://localhost:11434");
  });

  it("uses OLLAMA_HOST with http:// prefix", () => {
    process.env.OLLAMA_HOST = "http://my-server:11434";
    expect(getOllamaHost()).toBe("http://my-server:11434");
  });

  it("uses OLLAMA_HOST without prefix (adds http://)", () => {
    process.env.OLLAMA_HOST = "my-server:11434";
    expect(getOllamaHost()).toBe("http://my-server:11434");
  });

  it("uses OLLAMA_HOST with https:// prefix", () => {
    process.env.OLLAMA_HOST = "https://remote:11434";
    expect(getOllamaHost()).toBe("https://remote:11434");
  });

  it("strips trailing slash", () => {
    process.env.OLLAMA_HOST = "http://my-server:11434/";
    expect(getOllamaHost()).toBe("http://my-server:11434");
  });

  it("supports custom port", () => {
    process.env.OLLAMA_HOST = "http://192.168.1.100:8080";
    expect(getOllamaHost()).toBe("http://192.168.1.100:8080");
  });
});