/**
 * Extension settings, slash commands, and Ollama lifecycle helpers.
 *
 * Settings are stored in ~/.pi/agent/settings.json under the "ollama" key.
 * This avoids needing a separate config file while giving users control
 * over Ollama-specific options (temperature, num_ctx, top_p, etc.).
 *
 * Commands:
 *   /ollama-setup   — Interactive setup wizard (local or cloud)
 *   /ollama-refresh  — Re-discover models
 *   /ollama-pull     — Pull a model with progress bar
 *   /ollama-status   — Check Ollama connection status and loaded models
 *   /ollama-info     — Show detailed model info from /api/show
 */

import type { ExtensionAPI, ExtensionCommandContext } from "@mariozechner/pi-coding-agent";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

const SETTINGS_PATH = join(homedir(), ".pi", "agent", "settings.json");

// ── settings ──

export interface OllamaSettings {
  /** Default num_ctx for all models (default: model's context_length from /api/show) */
  defaultNumCtx?: number;
  /** Default keep_alive (default: "30m") */
  keepAlive?: string;
  /** Ollama options applied to every request */
  options?: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    num_predict?: number;
    repeat_penalty?: number;
    [key: string]: unknown;
  };
  /** Whether to use native /api/chat or OpenAI compat shim (default: "native") */
  streamingMode?: "native" | "openai-compat";
  /** Whether to auto-pull models on select (default: true) */
  autoPull?: boolean;
}

const DEFAULT_SETTINGS: OllamaSettings = {
  keepAlive: "30m",
  streamingMode: "native",
  autoPull: true,
};

/**
 * Read ollama settings from ~/.pi/agent/settings.json.
 * Merges with defaults for any missing keys.
 */
export function readSettings(): OllamaSettings {
  try {
    if (!existsSync(SETTINGS_PATH)) return { ...DEFAULT_SETTINGS };
    const data = readFileSync(SETTINGS_PATH, "utf-8");
    const parsed = JSON.parse(data);
    const ollamaSection = parsed?.ollama ?? {};
    return { ...DEFAULT_SETTINGS, ...ollamaSection };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

/**
 * Write ollama settings, preserving other settings in the file.
 */
export function writeSettings(settings: OllamaSettings): void {
  try {
    let existing: Record<string, unknown> = {};
    try {
      if (existsSync(SETTINGS_PATH)) {
        const data = readFileSync(SETTINGS_PATH, "utf-8");
        existing = JSON.parse(data);
      }
    } catch {}

    existing.ollama = settings;
    writeFileSync(SETTINGS_PATH, JSON.stringify(existing, null, 2), "utf-8");
  } catch (err) {
    console.log(
      `[ollama] settings write failed: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

// ── command: /ollama-status ──

export async function handleStatusCommand(
  pi: ExtensionAPI,
  ctx: ExtensionCommandContext,
  localBaseUrl: string,
  apiKey: string,
  localModelNames: Set<string>,
  loadedModels: string[],
): Promise<void> {
  ctx.ui.setStatus("ollama-status", "Checking Ollama status...");

  const lines: string[] = ["🦙 **Ollama Status**\n"];

  // Local server check
  try {
    const res = await fetch(`${localBaseUrl}/api/tags`, {
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(5000),
    });
    if (res.ok) {
      const data = await res.json();
      const modelCount = (data.models || []).length;
      lines.push(`✅ Local server: **running** at \`${localBaseUrl}\``);
      lines.push(`   ${modelCount} model(s) pulled`);

      if (loadedModels.length > 0) {
        lines.push(`   Currently loaded: ${loadedModels.join(", ")}`);
      } else {
        lines.push(`   No models currently loaded`);
      }
    } else {
      lines.push(`❌ Local server: **HTTP ${res.status}** at \`${localBaseUrl}\``);
    }
  } catch {
    lines.push(`❌ Local server: **not running** at \`${localBaseUrl}\``);
    lines.push(`   Start with: \`ollama serve\``);
  }

  // Cloud connectivity check (if API key is configured)
  if (apiKey && apiKey !== "ollama") {
    try {
      const res = await fetch("https://ollama.com/v1/models", {
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) {
        const data = await res.json();
        const modelCount = (data.data || []).length;
        lines.push(`✅ Cloud: **connected** (${modelCount} models available)`);
      } else {
        lines.push(`❌ Cloud: **HTTP ${res.status}** (check API key)`);
      }
    } catch {
      lines.push(`❌ Cloud: **unreachable**`);
    }
  } else {
    lines.push(`⚪ Cloud: not configured (run /ollama-setup)`);
  }

  // Settings summary
  const settings = readSettings();
  lines.push(
    `\n⚙️ **Settings**: streaming=${settings.streamingMode || "native"}, ` +
    `keepAlive=${settings.keepAlive || "30m"}, ` +
    `autoPull=${settings.autoPull !== false ? "yes" : "no"}`,
  );

  if (settings.options && Object.keys(settings.options).length > 0) {
    const optStrs = Object.entries(settings.options)
      .map(([k, v]) => `${k}=${v}`)
      .join(", ");
    lines.push(`   Options: ${optStrs}`);
  }

  ctx.ui.setStatus("ollama-status", undefined);
  ctx.ui.notify(lines.join("\n"), "info");
}

// ── command: /ollama-info ──

export async function handleInfoCommand(
  args: unknown,
  ctx: ExtensionCommandContext,
  localBaseUrl: string,
  apiKey?: string,
): Promise<void> {
  const modelName = typeof args === "string" ? args.trim() : args?.[0] ?? args?.model;

  if (!modelName) {
    ctx.ui.notify("Usage: /ollama-info <model-name>", "warning");
    return;
  }

  ctx.ui.setStatus("ollama-info", `Fetching info for ${modelName}...`);

  try {
    const res = await fetch(`${localBaseUrl}/api/show`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey && apiKey !== "ollama"
          ? { Authorization: `Bearer ${apiKey}` }
          : {}),
      },
      body: JSON.stringify({ name: modelName }),
    });

    if (!res.ok) {
      ctx.ui.notify(
        `✗ Model "${modelName}" not found (HTTP ${res.status})`,
        "error",
      );
      return;
    }

    const data = await res.json();
    const lines: string[] = [`🦙 **${modelName}**\n`];

    if (data.details) {
      if (data.details.family)
        lines.push(`- **Family**: ${data.details.family}`);
      if (data.details.parameter_size)
        lines.push(`- **Parameters**: ${data.details.parameter_size}`);
      if (data.details.quantization_level)
        lines.push(`- **Quantization**: ${data.details.quantization_level}`);
      if (data.details.capabilities)
        lines.push(`- **Capabilities**: ${data.details.capabilities.join(", ")}`);
    }

    if (data.model_info) {
      const ctxKeys = Object.keys(data.model_info).filter((k) =>
        k.endsWith(".context_length"),
      );
      if (ctxKeys.length > 0) {
        lines.push(
          `- **Context length**: ${data.model_info[ctxKeys[0]].toLocaleString()} tokens`,
        );
      }

      const archKey = Object.keys(data.model_info).find((k) =>
        k === "general.architecture",
      );
      if (archKey) {
        lines.push(`- **Architecture**: ${data.model_info[archKey]}`);
      }
    }

    if (data.template) {
      lines.push(`\n<details><summary>Chat template</summary>\n\n\`\`\`\n${data.template}\n\`\`\`\n</details>`);
    }

    if (data.system) {
      lines.push(`\n<details><summary>System prompt</summary>\n\n\`\`\`\n${data.system}\n\`\`\`\n</details>`);
    }

    if (data.parameters) {
      lines.push(`\n<details><summary>Model parameters</summary>\n\n\`\`\`\n${data.parameters}\n\`\`\`\n</details>`);
    }

    ctx.ui.setStatus("ollama-info", undefined);
    ctx.ui.notify(lines.join("\n"), "info");
  } catch (err) {
    ctx.ui.setStatus("ollama-info", undefined);
    ctx.ui.notify(
      `Error: ${err instanceof Error ? err.message : String(err)}`,
      "error",
    );
  }
}

// ── command: /ollama-pull ──

let pullingModels = new Set<string>();

export async function handlePullCommand(
  args: unknown,
  ctx: ExtensionCommandContext,
  baseUrl: string,
  apiKey: string,
): Promise<Set<string>> {
  const name =
    typeof args === "string"
      ? args.trim()
      : (args as any)?.model || (args as any)?.[0];

  if (!name) {
    ctx.ui.notify("Usage: /ollama-pull <model-name>", "error");
    return pullingModels;
  }

  if (pullingModels.has(name)) {
    ctx.ui.notify(`⏳ ${name} is already being pulled...`, "info");
    while (pullingModels.has(name))
      await new Promise((r) => setTimeout(r, 500));
    return pullingModels;
  }

  pullingModels.add(name);

  try {
    // Check if already pulled
    const checkRes = await fetch(`${baseUrl}/api/tags`, {
      headers: { "Content-Type": "application/json" },
    });
    if (checkRes.ok) {
      const data = await checkRes.json();
      if ((data.models || []).some((m: any) => m.name === name)) {
        ctx.ui.notify(`✓ ${name} is already pulled`, "info");
        return pullingModels;
      }
    }

    // Pull with progress
    const progressId = `ollama-pull-${name.replace(/[^a-z0-9]/gi, "-")}`;
    ctx.ui.setStatus(progressId, `⬇ Pulling ${name}...`);

    const pullRes = await fetch(`${baseUrl}/api/pull`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey && apiKey !== "ollama"
          ? { Authorization: `Bearer ${apiKey}` }
          : {}),
      },
      body: JSON.stringify({ name, stream: true }),
    });

    if (!pullRes.ok && !pullRes.body) throw new Error(`HTTP ${pullRes.status}`);

    const reader = pullRes.body!.getReader();
    const decoder = new TextDecoder();
    let lastProgress = 0;
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop()!;

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const event = JSON.parse(line);
          if (event.error) {
            throw new Error(event.error);
          } else if (event.status === "success") {
            lastProgress = 100;
            ctx.ui.setStatus(progressId, `✓ ${name} ready!`);
          } else if (event.total && event.completed && event.total > 0) {
            const pct = Math.round((event.completed / event.total) * 100);
            if (pct > lastProgress) {
              lastProgress = pct;
              const filled = Math.floor(pct / 5);
              ctx.ui.setStatus(
                progressId,
                `⬇ [${"█".repeat(filled)}${"░".repeat(20 - filled)}] ${pct}% ${name}`,
              );
            }
          } else if (event.status && lastProgress === 0) {
            ctx.ui.setStatus(progressId, `⬇ ${event.status} ${name}`);
          }
        } catch (e) {
          if (e instanceof Error) throw e;
        }
      }
    }

    pullingModels.add(name);
    ctx.ui.notify(`✓ ${name} pulled successfully!`, "info");
    setTimeout(() => ctx.ui.setStatus(progressId, undefined), 4000);
  } catch (err) {
    const progressId = `ollama-pull-${name.replace(/[^a-z0-9]/gi, "-")}`;
    ctx.ui.notify(
      `✗ ${name}: ${err instanceof Error ? err.message : String(err)}`,
      "error",
    );
    setTimeout(() => ctx.ui.setStatus(progressId, undefined), 5000);
    throw err;
  } finally {
    pullingModels.delete(name);
  }

  return pullingModels;
}

// ── setup wizard (exported for testing) ──

export interface SetupWizardDeps {
  localBaseUrl: string;
  cloudBaseUrl: string;
  apiKey: string;
  authStorage: {
    has(key: string): boolean;
    get(key: string): any;
    set(key: string, value: any): void;
    remove(key: string): void;
  };
}

export async function runSetupWizard(
  pi: ExtensionAPI,
  ctx: ExtensionCommandContext,
  deps: SetupWizardDeps,
  onConfigChange: (mode: "local" | "cloud", baseUrl: string, apiKey: string) => Promise<void>,
): Promise<void> {
  const { localBaseUrl, cloudBaseUrl, authStorage } = deps;

  const mode = await ctx.ui.select(
    "🦙 Ollama Setup — How would you like to use Ollama?",
    [
      "Local — Run models on this machine (requires Ollama CLI)",
      "Cloud — Use cloud models on ollama.com",
    ],
  );

  if (!mode) return;

  if (mode.startsWith("Local")) {
    const ready = await ensureOllamaCli(pi, ctx);
    if (!ready) return;

    if (authStorage.has("ollama")) {
      authStorage.remove("ollama");
    }

    await onConfigChange("local", localBaseUrl, "ollama");
    ctx.ui.notify(
      "✓ Setup complete! Use /models to see available Ollama models.",
      "info",
    );
    return;
  }

  // Cloud setup
  const authMethod = await ctx.ui.select(
    "🦙 Cloud Authentication — How would you like to authenticate?",
    [
      "API key — Enter an API key from ollama.com/settings/keys",
      "Browser login — Install Ollama CLI and run 'ollama signin'",
    ],
  );

  if (!authMethod) return;

  if (authMethod.startsWith("API key")) {
    const apiKey = await ctx.ui.input(
      "Enter your Ollama API key (from ollama.com/settings/keys):",
      "",
    );

    if (!apiKey) {
      ctx.ui.notify("Setup cancelled — no API key provided.", "warning");
      return;
    }

    ctx.ui.setStatus("ollama-setup", "Testing cloud connection...");
    try {
      const res = await fetch(`${cloudBaseUrl}/api/tags`, {
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
      });

      ctx.ui.setStatus("ollama-setup", undefined);

      if (!res.ok) {
        const body = await res.text().catch(() => "");
        ctx.ui.notify(
          `✗ Connection failed (HTTP ${res.status}): ${body}`,
          "error",
        );
        return;
      }

      const data = await res.json();
      const modelCount = (data.models || []).length;

      authStorage.set("ollama", { type: "api_key", key: apiKey });

      await onConfigChange("cloud", cloudBaseUrl, apiKey);
      ctx.ui.notify(
        `✓ Connected to Ollama Cloud! ${modelCount} models available.`,
        "info",
      );
    } catch (err) {
      ctx.ui.setStatus("ollama-setup", undefined);
      ctx.ui.notify(
        `✗ Connection failed: ${err instanceof Error ? err.message : String(err)}`,
        "error",
      );
    }
    return;
  }

  // Browser login
  const ready = await ensureOllamaCli(pi, ctx);
  if (!ready) return;

  ctx.ui.notify(
    "Running 'ollama signin'... Follow the browser prompts to authorize.",
    "info",
  );
  ctx.ui.setStatus("ollama-setup", "Waiting for ollama signin...");
  try {
    const result = await pi.exec("ollama", ["signin"], { timeout: 120000 });
    ctx.ui.setStatus("ollama-setup", undefined);
    if (result.code === 0) {
      ctx.ui.notify(
        "✓ Signed in! Cloud models are now available through your local Ollama.",
        "info",
      );
    } else {
      ctx.ui.notify(
        `Sign-in issue: ${result.stderr || result.stdout}`,
        "warning",
      );
    }
  } catch (err) {
    ctx.ui.setStatus("ollama-setup", undefined);
    ctx.ui.notify(
      `Sign-in error: ${err instanceof Error ? err.message : String(err)}`,
      "error",
    );
    return;
  }

  if (authStorage.has("ollama")) {
    authStorage.remove("ollama");
  }

  await onConfigChange("local", localBaseUrl, "ollama");
  ctx.ui.notify(
    "✓ Setup complete! Use /models to see available Ollama models.",
    "info",
  );
}

// ── setup helpers ──

async function checkOllamaInstalled(pi: ExtensionAPI): Promise<boolean> {
  try {
    const result = await pi.exec("ollama", ["--version"], { timeout: 5000 });
    return result.code === 0;
  } catch {
    return false;
  }
}

async function installOllama(
  pi: ExtensionAPI,
  ctx: ExtensionCommandContext,
): Promise<boolean> {
  ctx.ui.setStatus("ollama-setup", "⬇ Installing Ollama...");
  try {
    const result = await pi.exec("bash", [
      "-c",
      "curl -fsSL https://ollama.com/install.sh | sh",
    ], { timeout: 120000 });
    ctx.ui.setStatus("ollama-setup", undefined);
    if (result.code === 0) {
      ctx.ui.notify("✓ Ollama installed successfully!", "info");
      return true;
    } else {
      ctx.ui.notify(`✗ Installation failed: ${result.stderr}`, "error");
      return false;
    }
  } catch (err) {
    ctx.ui.setStatus("ollama-setup", undefined);
    ctx.ui.notify(
      `✗ Installation failed: ${err instanceof Error ? err.message : String(err)}`,
      "error",
    );
    return false;
  }
}

async function startOllamaService(
  pi: ExtensionAPI,
  ctx: ExtensionCommandContext,
): Promise<boolean> {
  ctx.ui.setStatus("ollama-setup", "Starting Ollama...");
  try {
    pi.exec("bash", ["-c", "nohup ollama serve > /dev/null 2>&1 &"], {
      timeout: 5000,
    }).catch(() => {});

    for (let i = 0; i < 10; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      try {
        const res = await fetch("http://localhost:11434/api/tags", {
          signal: AbortSignal.timeout(3000),
        });
        if (res.ok) {
          ctx.ui.setStatus("ollama-setup", undefined);
          ctx.ui.notify("✓ Ollama is running!", "info");
          return true;
        }
      } catch {}
    }

    ctx.ui.setStatus("ollama-setup", undefined);
    ctx.ui.notify(
      "✗ Ollama did not start in time. Try running 'ollama serve' manually.",
      "error",
    );
    return false;
  } catch (err) {
    ctx.ui.setStatus("ollama-setup", undefined);
    ctx.ui.notify(
      `✗ Could not start Ollama: ${err instanceof Error ? err.message : String(err)}`,
      "error",
    );
    return false;
  }
}

async function ensureOllamaCli(
  pi: ExtensionAPI,
  ctx: ExtensionCommandContext,
): Promise<boolean> {
  const isInstalled = await checkOllamaInstalled(pi);

  if (!isInstalled) {
    const install = await ctx.ui.confirm(
      "Ollama not found",
      "Ollama CLI is not installed on this machine. Install it now?",
    );

    if (!install) {
      ctx.ui.notify(
        "Setup cancelled. Install Ollama manually: https://ollama.com/download",
        "warning",
      );
      return false;
    }

    const installed = await installOllama(pi, ctx);
    if (!installed) return false;
  }

  // Check if running
  try {
    const res = await fetch("http://localhost:11434/api/tags", {
      signal: AbortSignal.timeout(3000),
    });
    if (res.ok) return true;
  } catch {}

  const start = await ctx.ui.confirm(
    "Ollama not running",
    "Ollama is installed but not running. Start it now?",
  );

  if (start) {
    return startOllamaService(pi, ctx);
  } else {
    ctx.ui.notify(
      "Start Ollama manually with 'ollama serve' before using models.",
      "warning",
    );
    return false;
  }
}