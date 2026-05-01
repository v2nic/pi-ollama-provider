/**
 * Tests for auth.ts — config resolution, auth header generation.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { writeFileSync, mkdirSync, rmSync, existsSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  resolveConfig,
  readOllamaAuthFromJson,
  authHeaders,
  DEFAULT_LOCAL_URL,
  DEFAULT_CLOUD_URL,
  type OllamaConfig,
} from "../extensions/pi-ollama-provider/auth.js";

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

// ════════════════════════════════════════════════════════════════
// readOllamaAuthFromJson
// ════════════════════════════════════════════════════════════════

describe("readOllamaAuthFromJson", () => {
  it("reads a valid api_key credential from 'ollama' key", () => {
    writeFileSync(authPath, JSON.stringify({ ollama: { type: "api_key", key: "my-key-123" } }));
    expect(readOllamaAuthFromJson(authPath)).toEqual({ type: "api_key", key: "my-key-123" });
  });

  it("reads a valid api_key credential from 'ollama-cloud' key", () => {
    writeFileSync(authPath, JSON.stringify({ "ollama-cloud": { type: "api_key", key: "cloud-key" } }));
    expect(readOllamaAuthFromJson(authPath)).toEqual({ type: "api_key", key: "cloud-key" });
  });

  it("prefers ollama-cloud over ollama", () => {
    writeFileSync(authPath, JSON.stringify({
      ollama: { type: "api_key", key: "old-key" },
      "ollama-cloud": { type: "api_key", key: "new-cloud-key" },
    }));
    expect(readOllamaAuthFromJson(authPath)).toEqual({ type: "api_key", key: "new-cloud-key" });
  });

  it("returns undefined when no ollama entry", () => {
    writeFileSync(authPath, JSON.stringify({ anthropic: { type: "api_key", key: "sk-ant" } }));
    expect(readOllamaAuthFromJson(authPath)).toBeUndefined();
  });

  it("returns undefined when file doesn't exist", () => {
    expect(readOllamaAuthFromJson(join(tempDir, "nope.json"))).toBeUndefined();
  });

  it("returns undefined for wrong credential type", () => {
    writeFileSync(authPath, JSON.stringify({ ollama: { type: "oauth", access: "token" } }));
    expect(readOllamaAuthFromJson(authPath)).toBeUndefined();
  });

  it("returns undefined for missing key field", () => {
    writeFileSync(authPath, JSON.stringify({ ollama: { type: "api_key" } }));
    expect(readOllamaAuthFromJson(authPath)).toBeUndefined();
  });

  it("returns undefined for malformed JSON", () => {
    writeFileSync(authPath, "not-json{{{");
    expect(readOllamaAuthFromJson(authPath)).toBeUndefined();
  });
});

// ════════════════════════════════════════════════════════════════
// resolveConfig — priority chain
// ════════════════════════════════════════════════════════════════

describe("resolveConfig", () => {
  it("stored credential wins over env var", () => {
    writeFileSync(authPath, JSON.stringify({ ollama: { type: "api_key", key: "stored-key" } }));
    const config = resolveConfig({ authPath, envKey: "env-key" });
    expect(config.apiKey).toBe("stored-key");
    expect(config.mode).toBe("cloud");
    expect(config.baseUrl).toBe(DEFAULT_CLOUD_URL);
  });

  it("env var wins when no stored credential", () => {
    writeFileSync(authPath, JSON.stringify({}));
    const config = resolveConfig({ authPath, envKey: "env-key-123" });
    expect(config.apiKey).toBe("env-key-123");
    expect(config.mode).toBe("cloud");
    expect(config.baseUrl).toBe(DEFAULT_CLOUD_URL);
  });

  it("defaults to local when nothing configured", () => {
    writeFileSync(authPath, JSON.stringify({}));
    const config = resolveConfig({ authPath });
    expect(config.apiKey).toBe("ollama");
    expect(config.mode).toBe("local");
    expect(config.baseUrl).toBe(DEFAULT_LOCAL_URL);
  });

  it("defaults to local when file missing", () => {
    const config = resolveConfig({ authPath: join(tempDir, "nope.json") });
    expect(config.apiKey).toBe("ollama");
    expect(config.mode).toBe("local");
  });

  it("env var used when auth.json has no ollama entry", () => {
    writeFileSync(authPath, JSON.stringify({ anthropic: { type: "api_key", key: "sk-ant" } }));
    const config = resolveConfig({ authPath, envKey: "env-ollama-key" });
    expect(config.apiKey).toBe("env-ollama-key");
    expect(config.mode).toBe("cloud");
  });

  it("stored key='ollama' → local mode", () => {
    writeFileSync(authPath, JSON.stringify({ ollama: { type: "api_key", key: "ollama" } }));
    const config = resolveConfig({ authPath });
    expect(config.apiKey).toBe("ollama");
    expect(config.mode).toBe("local");
    expect(config.baseUrl).toBe(DEFAULT_LOCAL_URL);
  });

  it("no env var, no stored → local", () => {
    const config = resolveConfig({ authPath, envKey: undefined });
    expect(config.mode).toBe("local");
  });

  it("empty string env var is ignored", () => {
    const config = resolveConfig({ authPath, envKey: "" });
    expect(config.apiKey).toBe("ollama");
    expect(config.mode).toBe("local");
  });

  it("uses real process.env.OLLAMA_API_KEY when envKey not specified", () => {
    process.env.OLLAMA_API_KEY = "from-real-env";
    writeFileSync(authPath, JSON.stringify({}));
    const config = resolveConfig({ authPath });
    expect(config.apiKey).toBe("from-real-env");
    expect(config.mode).toBe("cloud");
  });
});

// ════════════════════════════════════════════════════════════════
// authHeaders
// ════════════════════════════════════════════════════════════════

describe("authHeaders", () => {
  it("includes Bearer header for cloud apiKey", () => {
    const headers = authHeaders({ mode: "cloud", baseUrl: DEFAULT_CLOUD_URL, apiKey: "my-key" });
    expect(headers["Authorization"]).toBe("Bearer my-key");
    expect(headers["Content-Type"]).toBe("application/json");
  });

  it("omits Bearer header for default 'ollama' key", () => {
    const headers = authHeaders({ mode: "local", baseUrl: DEFAULT_LOCAL_URL, apiKey: "ollama" });
    expect(headers["Authorization"]).toBeUndefined();
    expect(headers["Content-Type"]).toBe("application/json");
  });

  it("omits Bearer header for empty string key", () => {
    const headers = authHeaders({ mode: "local", baseUrl: DEFAULT_LOCAL_URL, apiKey: "" });
    expect(headers["Authorization"]).toBeUndefined();
  });
});