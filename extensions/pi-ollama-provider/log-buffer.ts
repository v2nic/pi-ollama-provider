/**
 * Structured log buffer for the Ollama extension.
 *
 * Replaces console.log/error calls that pollute the Pi TUI with a
 * ring buffer of structured entries. Errors and warnings are surfaced
 * via the Pi status bar (using ctx.ui.setStatus) and the /ollama-status
 * command, keeping the conversation view clean.
 *
 * During startup, only `pi` (ExtensionAPI) is available — no UI context.
 * Errors are buffered here and displayed once `session_start` fires
 * (which provides `ctx` with `ctx.ui.setStatus`).
 */

// ── types ──

export type LogLevel = "warn" | "error";

export interface LogEntry {
  /** Unix timestamp (Date.now()) */
  timestamp: number;
  /** Severity level */
  level: LogLevel;
  /** Which subsystem produced this entry */
  source: "local" | "cloud" | "cache" | "settings" | "stream" | "auto-pull";
  /** Human-readable message */
  message: string;
}

// ── ring buffer ──

const MAX_ENTRIES = 20;

const logBuffer: LogEntry[] = [];

function addEntry(level: LogLevel, source: LogEntry["source"], message: string): void {
  logBuffer.push({ timestamp: Date.now(), level, source, message });
  if (logBuffer.length > MAX_ENTRIES) {
    logBuffer.splice(0, logBuffer.length - MAX_ENTRIES);
  }
}

// ── public API ──

/** Log a warning. Shown in status bar (auto-cleared) and /ollama-status. */
export function logWarn(source: LogEntry["source"], message: string): void {
  addEntry("warn", source, message);
}

/** Log an error. Shown in status bar (persistent) and /ollama-status. */
export function logError(source: LogEntry["source"], message: string): void {
  addEntry("error", source, message);
}

/** Return all buffered entries (newest last). */
export function getLogEntries(): LogEntry[] {
  return [...logBuffer];
}

/** Return the N most recent entries (newest last). */
export function getRecentEntries(count: number = 5): LogEntry[] {
  return logBuffer.slice(-count);
}

/** Return only error-level entries. */
export function getErrors(): LogEntry[] {
  return logBuffer.filter((e) => e.level === "error");
}

/** Clear all buffered entries. */
export function clearLog(): void {
  logBuffer.length = 0;
}

/**
 * Format a one-line status string for the Pi footer/status bar.
 *
 * Returns `undefined` if there are no errors or warnings to show.
 *
 * Examples:
 *   "⚠ ollama: local server unreachable"
 *   "⚠ ollama: 2 errors — local unreachable, cloud auth failed"
 *   "⚠ ollama: 4 errors — local unreachable, cloud auth failed, +2 more"
 */
export function formatStatusLine(): string | undefined {
  const entries = getRecentEntries(3);
  if (entries.length === 0) return undefined;

  const errors = entries.filter((e) => e.level === "error");
  const warnings = entries.filter((e) => e.level === "warn");

  if (errors.length === 0 && warnings.length === 0) return undefined;

  if (errors.length === 1 && warnings.length === 0) {
    return `⚠ ollama: ${errors[0].message}`;
  }

  if (errors.length === 0 && warnings.length === 1) {
    return `⚠ ollama: ${warnings[0].message}`;
  }

  // Multiple entries — summarize
  const allRecent = entries;
  const totalErrors = getErrors().length;
  const totalEntries = logBuffer.length;
  const overflow = totalEntries - allRecent.length;

  const summaries = allRecent.map((e) => e.message);
  const suffix = overflow > 0 ? `, +${overflow} more` : "";

  return `⚠ ollama: ${totalErrors} error${totalErrors !== 1 ? "s" : ""} — ${summaries.join(", ")}${suffix}`;
}