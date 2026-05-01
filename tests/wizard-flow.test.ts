import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

// ── import from the real module ──
import {
  runSetupWizard,
  DEFAULT_CLOUD_URL,
} from "../extensions/pi-ollama-provider/index.js";

import type { ExtensionAPI, ExtensionCommandContext } from "@mariozechner/pi-coding-agent";

// ── mock builders ──

/** Build a mock AuthStorage that records calls. */
function mockAuthStore() {
  const store: Record<string, any> = {};
  const log: { method: string; key: string; value?: any }[] = [];
  return {
    has: vi.fn((key: string) => {
      log.push({ method: "has", key });
      return key in store;
    }),
    get: vi.fn((key: string) => {
      log.push({ method: "get", key });
      return store[key];
    }),
    set: vi.fn((key: string, value: any) => {
      log.push({ method: "set", key, value });
      store[key] = value;
    }),
    remove: vi.fn((key: string) => {
      log.push({ method: "remove", key });
      delete store[key];
    }),
    log,
  };
}

/** Build a mock UI context with scripted responses. */
function mockUI(responses: {
  selects?: string[];
  confirms?: boolean[];
  inputs?: string[];
}) {
  let selectIdx = 0;
  let confirmIdx = 0;
  let inputIdx = 0;

  const notifications: { message: string; type?: string }[] = [];
  const statuses: { key: string; text: string | undefined }[] = [];

  return {
    select: vi.fn(async () => responses.selects?.[selectIdx++] ?? undefined),
    confirm: vi.fn(async () => responses.confirms?.[confirmIdx++] ?? false),
    input: vi.fn(async () => responses.inputs?.[inputIdx++] ?? undefined),
    notify: vi.fn((message: string, type?: string) => notifications.push({ message, type })),
    setStatus: vi.fn((key: string, text: string | undefined) => statuses.push({ key, text })),
    notifications,
    statuses,
  };
}

/** Build mock pi with exec that reports ollama as installed & running. */
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

/** Build a full mock ctx. */
function mockCtx(opts: {
  authStore?: ReturnType<typeof mockAuthStore>;
  ui?: ReturnType<typeof mockUI>;
  fetchResponse?: { ok: boolean; status: number; json?: any; text?: string };
}) {
  const authStore = opts.authStore ?? mockAuthStore();
  const ui = opts.ui ?? mockUI({});

  const ctx = {
    ui,
    hasUI: true,
    modelRegistry: { authStorage: authStore },
  } as unknown as ExtensionCommandContext;

  // Mock global fetch
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.includes("/api/tags")) {
      // checkOllamaRunning uses localhost — return ok for local
      if (url.includes("localhost")) {
        return { ok: true, json: async () => ({ models: [] }) } as Response;
      }
      // Cloud test connection
      const resp = opts.fetchResponse ?? { ok: true, status: 200, json: { models: [{ name: "qwen3:8b", size: 4e9 }] } };
      if (resp.ok) {
        return { ok: true, status: 200, json: async () => resp.json } as Response;
      }
      return { ok: false, status: resp.status, text: async () => resp.text ?? "" } as Response;
    }
    // fetchModelDetails
    return { ok: true, json: async () => ({ details: { capabilities: [] }, model_info: {} }) } as Response;
  });

  return { ctx, authStore, ui, fetchMock };
}

// ════════════════════════════════════════════════════════════════
// Wizard flow tests
// ════════════════════════════════════════════════════════════════

describe("runSetupWizard", () => {
  let fetchSpy: any;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  // ── Local mode ──

  it("Local: clears stored cloud credential and sets local mode", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    // Pre-populate a cloud credential
    store.set("ollama", { type: "api_key", key: "old-cloud-key" });
    store.log.length = 0; // clear setup log

    const ui = mockUI({ selects: ["Local"], confirms: [true] });
    const { ctx } = mockCtx({ authStore: store, ui });

    fetchSpy.mockImplementation(async (url: string) => {
      if (url.includes("localhost")) return { ok: true, json: async () => ({ models: [] }) } as Response;
      return { ok: true, json: async () => ({ models: [] }) } as Response;
    });

    await runSetupWizard(pi, ctx);

    // Should have removed the old credential
    expect(store.remove).toHaveBeenCalledWith("ollama");
    expect(ui.notifications.some((n) => n.message.includes("Setup complete"))).toBe(true);
  });

  // ── Cloud + API key (success) ──

  it("Cloud + API key: saves to authStorage on success", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Cloud", "API key"], inputs: ["my-api-key"] });
    const { ctx } = mockCtx({
      authStore: store,
      ui,
      fetchResponse: { ok: true, status: 200, json: { models: [{ name: "qwen3:8b", size: 4e9 }] } },
    });

    fetchSpy.mockImplementation(async (url: string, init?: RequestInit) => {
      if (url.includes("localhost")) return { ok: true, json: async () => ({ models: [] }) } as Response;
      return { ok: true, status: 200, json: async () => ({ models: [{ name: "qwen3:8b", size: 4e9 }] }) } as Response;
    });

    await runSetupWizard(pi, ctx);

    expect(store.set).toHaveBeenCalledWith("ollama", { type: "api_key", key: "my-api-key" });
    expect(ui.notifications.some((n) => n.message.includes("Connected to Ollama Cloud"))).toBe(true);
  });

  // ── Cloud + API key (connection failure) ──

  it("Cloud + API key: does NOT save on HTTP failure", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Cloud", "API key"], inputs: ["bad-key"] });

    fetchSpy.mockImplementation(async (url: string) => {
      if (url.includes("localhost")) return { ok: true, json: async () => ({ models: [] }) } as Response;
      return { ok: false, status: 401, text: async () => "Unauthorized" } as Response;
    });

    const { ctx } = mockCtx({ authStore: store, ui });

    await runSetupWizard(pi, ctx);

    expect(store.set).not.toHaveBeenCalledWith(
      "ollama",
      expect.objectContaining({ key: "bad-key" }),
    );
    expect(ui.notifications.some((n) => n.message.includes("Connection failed"))).toBe(true);
  });

  // ── Cloud + API key (empty input = cancelled) ──

  it("Cloud + API key: cancelled when user enters nothing", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Cloud", "API key"], inputs: [""] });
    const { ctx } = mockCtx({ authStore: store, ui });

    fetchSpy.mockImplementation(async (url: string) => {
      return { ok: true, json: async () => ({ models: [] }) } as Response;
    });

    await runSetupWizard(pi, ctx);

    expect(store.set).not.toHaveBeenCalled();
    expect(ui.notifications.some((n) => n.message.includes("cancelled"))).toBe(true);
  });

  // ── Cloud + Browser login (success) ──

  it("Cloud + Browser login: clears stored key and sets local mode", async () => {
    const pi = mockPi({ ollamaInstalled: true, ollamaSigninCode: 0 });
    const store = mockAuthStore();
    store.set("ollama", { type: "api_key", key: "old-key" });
    store.log.length = 0;

    const ui = mockUI({ selects: ["Cloud", "Browser login"], confirms: [true] });

    fetchSpy.mockImplementation(async (url: string) => {
      if (url.includes("localhost")) return { ok: true, json: async () => ({ models: [] }) } as Response;
      return { ok: true, json: async () => ({ models: [] }) } as Response;
    });

    const { ctx } = mockCtx({ authStore: store, ui });

    await runSetupWizard(pi, ctx);

    expect(store.remove).toHaveBeenCalledWith("ollama");
    expect(ui.notifications.some((n) => n.message.includes("Setup complete") || n.message.includes("Signed in"))).toBe(true);
  });

  // ── Cancel at mode selection ──

  it("Cancel at step 1: no changes", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: [undefined] }); // Escape
    const { ctx } = mockCtx({ authStore: store, ui });

    fetchSpy.mockImplementation(async () => {
      return { ok: true, json: async () => ({ models: [] }) } as Response;
    });

    await runSetupWizard(pi, ctx);

    expect(store.set).not.toHaveBeenCalled();
    expect(store.remove).not.toHaveBeenCalled();
  });

  // ── Cancel at auth method selection ──

  it("Cancel at step 2: no changes", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Cloud", undefined] }); // Escape at auth method
    const { ctx } = mockCtx({ authStore: store, ui });

    fetchSpy.mockImplementation(async () => {
      return { ok: true, json: async () => ({ models: [] }) } as Response;
    });

    await runSetupWizard(pi, ctx);

    expect(store.set).not.toHaveBeenCalled();
  });
});