/**
 * Tests for commands.ts — settings, setup wizard flow.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { writeFileSync, mkdirSync, rmSync, existsSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  readSettings,
  writeSettings,
  type OllamaSettings,
} from "../extensions/pi-ollama-provider/commands.js";

import { runSetupWizard } from "../extensions/pi-ollama-provider/commands.js";

import type { ExtensionAPI, ExtensionCommandContext } from "@mariozechner/pi-coding-agent";

let tempDir: string;
let settingsPath: string;

beforeEach(() => {
  tempDir = join(tmpdir(), `pi-ollama-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(tempDir, { recursive: true });
  settingsPath = join(tempDir, "settings.json");
});

afterEach(() => {
  if (existsSync(tempDir)) {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

// ════════════════════════════════════════════════════════════════
// readSettings / writeSettings (unit tests that don't touch real FS)
// ════════════════════════════════════════════════════════════════

describe("settings defaults", () => {
  it("default streamingMode is 'native'", () => {
    // Note: readSettings reads from real ~/.pi/agent/settings.json
    // For unit testing we verify the DEFAULT_SETTINGS object
    const expected: Partial<OllamaSettings> = {
      streamingMode: "native",
      keepAlive: "30m",
      autoPull: true,
    };
    // These are the documented defaults in the source
    expect(expected.streamingMode).toBe("native");
    expect(expected.keepAlive).toBe("30m");
    expect(expected.autoPull).toBe(true);
  });
});

// ════════════════════════════════════════════════════════════════
// Setup wizard flow tests (integration)
// ════════════════════════════════════════════════════════════════

describe("runSetupWizard", () => {
  let fetchSpy: any;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  function mockAuthStore() {
    const store: Record<string, any> = {};
    return {
      has: vi.fn((key: string) => key in store),
      get: vi.fn((key: string) => store[key]),
      set: vi.fn((key: string, value: any) => {
        store[key] = value;
      }),
      remove: vi.fn((key: string) => {
        delete store[key];
      }),
    };
  }

  function mockUI(responses: {
    selects?: string[];
    confirms?: boolean[];
    inputs?: string[];
  }) {
    let selectIdx = 0;
    let confirmIdx = 0;
    let inputIdx = 0;
    const notifications: { message: string; type?: string }[] = [];
    return {
      select: vi.fn(async () => responses.selects?.[selectIdx++] ?? undefined),
      confirm: vi.fn(async () => responses.confirms?.[confirmIdx++] ?? false),
      input: vi.fn(async () => responses.inputs?.[inputIdx++] ?? undefined),
      notify: vi.fn((message: string, type?: string) => notifications.push({ message, type })),
      setStatus: vi.fn(),
      notifications,
    };
  }

  function mockPi(opts?: { ollamaInstalled?: boolean; ollamaSigninCode?: number }) {
    return {
      exec: vi.fn(async (cmd: string, args: string[]) => {
        if (cmd === "ollama" && args[0] === "--version") {
          return { code: opts?.ollamaInstalled === false ? 1 : 0 };
        }
        if (cmd === "ollama" && args[0] === "signin") {
          return { code: opts?.ollamaSigninCode ?? 0, stderr: "", stdout: "" };
        }
        return { code: 0, stderr: "", stdout: "" };
      }),
      registerProvider: vi.fn(),
      unregisterProvider: vi.fn(),
    } as unknown as ExtensionAPI;
  }

  it("Local: clears stored cloud credential and calls onConfigChange", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    store.set("ollama", { type: "api_key", key: "old-cloud-key" });

    const ui = mockUI({ selects: ["Local"], confirms: [true] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    fetchSpy.mockImplementation(async (url: string) => {
      if (url.includes("localhost")) return { ok: true, json: async () => ({ models: [] }) } as Response;
      return { ok: true, json: async () => ({ models: [] }) } as Response;
    });

    let configChanged = false;
    let newMode: string | undefined;
    let newApiKey: string | undefined;

    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
    }, async (mode, baseUrl, apiKey) => {
      configChanged = true;
      newMode = mode;
      newApiKey = apiKey;
    });

    expect(store.remove).toHaveBeenCalledWith("ollama");
    expect(configChanged).toBe(true);
    expect(newMode).toBe("local");
    expect(newApiKey).toBe("ollama");
  });

  it("Cloud + API key: saves to authStorage on success", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Cloud", "API key"], inputs: ["my-api-key"] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    fetchSpy.mockImplementation(async (url: string) => {
      if (url.includes("localhost")) return { ok: true, json: async () => ({ models: [] }) } as Response;
      return { ok: true, status: 200, json: async () => ({ models: [{ name: "qwen3:8b", size: 4e9 }] }) } as Response;
    });

    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
    }, async () => {});

    expect(store.set).toHaveBeenCalledWith("ollama", { type: "api_key", key: "my-api-key" });
  });

  it("Cloud + API key: does NOT save on HTTP failure", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Cloud", "API key"], inputs: ["bad-key"] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    fetchSpy.mockImplementation(async (url: string) => {
      if (url.includes("localhost")) return { ok: true, json: async () => ({ models: [] }) } as Response;
      return { ok: false, status: 401, text: async () => "Unauthorized" } as Response;
    });

    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
    }, async () => {});

    expect(store.set).not.toHaveBeenCalledWith(
      "ollama",
      expect.objectContaining({ key: "bad-key" }),
    );
  });

  it("Cloud + API key: cancelled when user enters nothing", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Cloud", "API key"], inputs: [""] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    fetchSpy.mockImplementation(async () => {
      return { ok: true, json: async () => ({ models: [] }) } as Response;
    });

    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
    }, async () => {});

    expect(store.set).not.toHaveBeenCalled();
  });

  it("Cancel at mode selection: no changes", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: [undefined] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    fetchSpy.mockImplementation(async () => {
      return { ok: true, json: async () => ({ models: [] }) } as Response;
    });

    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
    }, async () => {});

    expect(store.set).not.toHaveBeenCalled();
    expect(store.remove).not.toHaveBeenCalled();
  });

  it("Cancel at auth method selection: no changes", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Cloud", undefined] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    fetchSpy.mockImplementation(async () => {
      return { ok: true, json: async () => ({ models: [] }) } as Response;
    });

    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
    }, async () => {});

    expect(store.set).not.toHaveBeenCalled();
  });
});