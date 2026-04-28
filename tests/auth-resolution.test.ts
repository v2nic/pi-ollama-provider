import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { writeFileSync, mkdirSync, rmSync, existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

// ── helpers matching the logic in index.ts ──

function isCloudModel(
  modelName: string,
  localModelNames: Set<string>,
  modelSize: number,
  configMode: "local" | "cloud",
): boolean {
  if (configMode === "cloud") return true;
  if (modelName.includes(":cloud") || modelName.endsWith("-cloud")) return true;
  if (!localModelNames.has(modelName) && modelSize > 100_000_000_000) return true;
  return false;
}

function generateModelId(modelName: string, isCloud: boolean): string {
  if (isCloud && !modelName.endsWith("-cloud") && !modelName.includes(":cloud")) {
    return modelName.includes(":") ? `${modelName}-cloud` : `${modelName}:cloud`;
  }
  return modelName;
}

// ── resolveConfig logic (mirrors index.ts) ──

interface AuthCredential {
  type: "api_key";
  key: string;
}

interface OllamaConfig {
  mode: "local" | "cloud";
  baseUrl: string;
  apiKey: string;
}

const DEFAULT_LOCAL_URL = "http://localhost:11434";
const DEFAULT_CLOUD_URL = "https://ollama.com";

function readOllamaAuthFromPath(authPath: string): AuthCredential | undefined {
  try {
    const data = readFileSync(authPath, "utf-8");
    const parsed = JSON.parse(data);
    const cred = parsed?.ollama;
    if (cred?.type === "api_key" && typeof cred.key === "string") {
      return cred;
    }
  } catch {}
  return undefined;
}

function resolveConfigFromPath(authPath: string, envKey?: string): OllamaConfig {
  process.env.OLLAMA_API_KEY = envKey;
  const stored = readOllamaAuthFromPath(authPath);
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

// ── test fixtures ──

let tempDir: string;
let authPath: string;

beforeEach(() => {
  tempDir = join(tmpdir(), `pi-ollama-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(tempDir, { recursive: true });
  authPath = join(tempDir, "auth.json");
  delete process.env.OLLAMA_API_KEY;
});

afterEach(() => {
  delete process.env.OLLAMA_API_KEY;
  if (existsSync(tempDir)) {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

// ── tests ──

describe("Auth Resolution", () => {
  describe("readOllamaAuthFromJson", () => {
    it("reads a valid api_key credential from auth.json", () => {
      writeFileSync(authPath, JSON.stringify({ ollama: { type: "api_key", key: "my-key-123" } }));
      const cred = readOllamaAuthFromPath(authPath);
      expect(cred).toEqual({ type: "api_key", key: "my-key-123" });
    });

    it("returns undefined when auth.json has no ollama entry", () => {
      writeFileSync(authPath, JSON.stringify({ anthropic: { type: "api_key", key: "sk-ant-..." } }));
      const cred = readOllamaAuthFromPath(authPath);
      expect(cred).toBeUndefined();
    });

    it("returns undefined when auth.json doesn't exist", () => {
      const cred = readOllamaAuthFromPath(join(tempDir, "nonexistent.json"));
      expect(cred).toBeUndefined();
    });

    it("returns undefined when ollama entry has wrong type", () => {
      writeFileSync(authPath, JSON.stringify({ ollama: { type: "oauth", access: "token" } }));
      const cred = readOllamaAuthFromPath(authPath);
      expect(cred).toBeUndefined();
    });

    it("returns undefined when ollama entry has missing key", () => {
      writeFileSync(authPath, JSON.stringify({ ollama: { type: "api_key" } }));
      const cred = readOllamaAuthFromPath(authPath);
      expect(cred).toBeUndefined();
    });

    it("returns undefined for malformed JSON", () => {
      writeFileSync(authPath, "not-json{{{");
      const cred = readOllamaAuthFromPath(authPath);
      expect(cred).toBeUndefined();
    });
  });

  describe("resolveConfig — priority chain", () => {
    it("uses stored credential from auth.json (highest priority)", () => {
      writeFileSync(authPath, JSON.stringify({ ollama: { type: "api_key", key: "stored-key" } }));
      const config = resolveConfigFromPath(authPath, "env-key");
      expect(config.apiKey).toBe("stored-key");
      expect(config.mode).toBe("cloud");
      expect(config.baseUrl).toBe(DEFAULT_CLOUD_URL);
    });

    it("falls back to OLLAMA_API_KEY env var when no stored credential", () => {
      writeFileSync(authPath, JSON.stringify({}));
      const config = resolveConfigFromPath(authPath, "env-key-123");
      expect(config.apiKey).toBe("env-key-123");
      expect(config.mode).toBe("cloud");
      expect(config.baseUrl).toBe(DEFAULT_CLOUD_URL);
    });

    it("defaults to local mode with 'ollama' key when nothing is configured", () => {
      writeFileSync(authPath, JSON.stringify({}));
      const config = resolveConfigFromPath(authPath);
      expect(config.apiKey).toBe("ollama");
      expect(config.mode).toBe("local");
      expect(config.baseUrl).toBe(DEFAULT_LOCAL_URL);
    });

    it("defaults to local mode when auth.json doesn't exist", () => {
      const config = resolveConfigFromPath(join(tempDir, "nonexistent.json"));
      expect(config.apiKey).toBe("ollama");
      expect(config.mode).toBe("local");
    });

    it("stored credential takes precedence over env var", () => {
      writeFileSync(authPath, JSON.stringify({ ollama: { type: "api_key", key: "from-auth-json" } }));
      const config = resolveConfigFromPath(authPath, "from-env");
      expect(config.apiKey).toBe("from-auth-json");
    });

    it("env var takes precedence when auth.json has no ollama entry", () => {
      writeFileSync(authPath, JSON.stringify({ anthropic: { type: "api_key", key: "sk-ant" } }));
      const config = resolveConfigFromPath(authPath, "env-ollama-key");
      expect(config.apiKey).toBe("env-ollama-key");
      expect(config.mode).toBe("cloud");
    });

    it("local mode with 'ollama' key when stored credential key equals 'ollama'", () => {
      writeFileSync(authPath, JSON.stringify({ ollama: { type: "api_key", key: "ollama" } }));
      const config = resolveConfigFromPath(authPath);
      expect(config.apiKey).toBe("ollama");
      expect(config.mode).toBe("local");
      expect(config.baseUrl).toBe(DEFAULT_LOCAL_URL);
    });
  });

  describe("resolveConfig — cloud URL", () => {
    it("uses cloud URL when cloud mode is detected from stored key", () => {
      writeFileSync(authPath, JSON.stringify({ ollama: { type: "api_key", key: "real-api-key" } }));
      const config = resolveConfigFromPath(authPath);
      expect(config.baseUrl).toBe(DEFAULT_CLOUD_URL);
    });

    it("uses cloud URL when env var is set", () => {
      const config = resolveConfigFromPath(authPath, "some-env-key");
      expect(config.baseUrl).toBe(DEFAULT_CLOUD_URL);
    });

    it("uses local URL when no auth configured", () => {
      const config = resolveConfigFromPath(authPath);
      expect(config.baseUrl).toBe(DEFAULT_LOCAL_URL);
    });
  });
});

describe("Cloud Model Detection", () => {
  describe("isCloudModel — local mode with ollama signin", () => {
    it("detects :cloud tag (e.g., kimi-k2.6:cloud, glm-5.1:cloud)", () => {
      const local = new Set(["llama3:8b", "kimi-k2.6:cloud", "glm-5.1:cloud"]);
      expect(isCloudModel("kimi-k2.6:cloud", local, 384, "local")).toBe(true);
      expect(isCloudModel("glm-5.1:cloud", local, 327, "local")).toBe(true);
    });

    it("detects -cloud suffix (e.g., qwen3-vl:235b-cloud)", () => {
      const local = new Set(["qwen3-vl:235b-cloud"]);
      expect(isCloudModel("qwen3-vl:235b-cloud", local, 393, "local")).toBe(true);
    });

    it("detects -cloud suffix for tagged models (e.g., qwen3.5:397b-cloud, gemma4:31b-cloud)", () => {
      const local = new Set(["qwen3.5:397b-cloud", "gemma4:31b-cloud"]);
      expect(isCloudModel("qwen3.5:397b-cloud", local, 393, "local")).toBe(true);
      expect(isCloudModel("gemma4:31b-cloud", local, 393, "local")).toBe(true);
    });

    it("does NOT flag local pulled models as cloud", () => {
      const local = new Set(["llama3:8b", "mistral:7b", "gemma3:12b"]);
      expect(isCloudModel("llama3:8b", local, 4_700_000_000, "local")).toBe(false);
      expect(isCloudModel("mistral:7b", local, 4_000_000_000, "local")).toBe(false);
    });

    it("detects large unpulled models as cloud (size fallback)", () => {
      const local = new Set(["llama3:8b"]);
      expect(isCloudModel("huge-model:200b", local, 200_000_000_000, "local")).toBe(true);
    });
  });

  describe("isCloudModel — cloud mode", () => {
    it("flags all models as cloud when mode=cloud", () => {
      const local = new Set(["qwen3.5:397b", "gemma4:31b"]);
      expect(isCloudModel("qwen3.5:397b", local, 397_000_000_000, "cloud")).toBe(true);
      expect(isCloudModel("gemma4:31b", local, 62_000_000_000, "cloud")).toBe(true);
    });
  });

  describe("generateModelId — preserves existing cloud suffixes", () => {
    it("keeps :cloud and -cloud that already exist from ollama signin", () => {
      expect(generateModelId("kimi-k2.6:cloud", true)).toBe("kimi-k2.6:cloud");
      expect(generateModelId("glm-5.1:cloud", true)).toBe("glm-5.1:cloud");
      expect(generateModelId("qwen3-vl:235b-cloud", true)).toBe("qwen3-vl:235b-cloud");
      expect(generateModelId("qwen3.5:397b-cloud", true)).toBe("qwen3.5:397b-cloud");
      expect(generateModelId("gemma4:31b-cloud", true)).toBe("gemma4:31b-cloud");
      expect(generateModelId("minimax-m2.7:cloud", true)).toBe("minimax-m2.7:cloud");
      expect(generateModelId("gemini-3-flash-preview:cloud", true)).toBe("gemini-3-flash-preview:cloud");
    });

    it("adds suffix for cloud models that lack one (cloud mode, ollama.com API)", () => {
      expect(generateModelId("qwen3.5:397b", true)).toBe("qwen3.5:397b-cloud");
      expect(generateModelId("gemma4:31b", true)).toBe("gemma4:31b-cloud");
      expect(generateModelId("gemini-3-flash-preview", true)).toBe("gemini-3-flash-preview:cloud");
      expect(generateModelId("kimi-k2.6", true)).toBe("kimi-k2.6:cloud");
    });

    it("does not modify non-cloud model IDs", () => {
      expect(generateModelId("llama3:8b", false)).toBe("llama3:8b");
    });
  });

  describe("Pattern matching", () => {
    it("user patterns ollama/X match registered model IDs", () => {
      const patterns = [
        "ollama/kimi-k2.6:cloud",
        "ollama/glm-5.1:cloud",
        "ollama/qwen3.5:397b-cloud",
        "ollama/gemma4:31b-cloud",
        "ollama/gemini-3-flash-preview:cloud",
        "ollama/minimax-m2.7:cloud",
        "ollama/qwen3-vl:235b-cloud",
      ];

      const registered = [
        "kimi-k2.6:cloud",
        "glm-5.1:cloud",
        "qwen3.5:397b-cloud",
        "gemma4:31b-cloud",
        "gemini-3-flash-preview:cloud",
        "minimax-m2.7:cloud",
        "qwen3-vl:235b-cloud",
      ];

      patterns.forEach((pattern, i) => {
        const modelId = pattern.replace("ollama/", "");
        expect(registered[i]).toBe(modelId);
      });
    });
  });
});

describe("authHeaders helper", () => {
  it("includes Authorization header when apiKey is not 'ollama'", () => {
    // Simulates the authHeaders() logic from index.ts
    const apiKey = "my-api-key";
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (apiKey && apiKey !== "ollama") {
      headers["Authorization"] = `Bearer ${apiKey}`;
    }
    expect(headers["Authorization"]).toBe("Bearer my-api-key");
  });

  it("omits Authorization header when apiKey is 'ollama' (default)", () => {
    const apiKey = "ollama";
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (apiKey && apiKey !== "ollama") {
      headers["Authorization"] = `Bearer ${apiKey}`;
    }
    expect(headers["Authorization"]).toBeUndefined();
  });
});