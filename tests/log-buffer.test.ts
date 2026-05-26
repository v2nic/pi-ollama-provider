/**
 * Tests for log-buffer.ts — structured logging for the Pi TUI status bar.
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  logWarn,
  logError,
  getLogEntries,
  getRecentEntries,
  getErrors,
  clearLog,
  formatStatusLine,
  type LogEntry,
} from "../extensions/pi-ollama-provider/log-buffer.js";

describe("log-buffer", () => {
  // Clear the log before each test to avoid cross-test contamination
  beforeEach(() => {
    clearLog();
  });

  // ── logWarn / logError ──

  describe("logWarn", () => {
    it("adds a warn entry to the buffer", () => {
      logWarn("local", "server unreachable");
      const entries = getLogEntries();
      expect(entries).toHaveLength(1);
      expect(entries[0].level).toBe("warn");
      expect(entries[0].source).toBe("local");
      expect(entries[0].message).toBe("server unreachable");
      expect(entries[0].timestamp).toBeGreaterThan(0);
    });

    it("adds entries from different sources", () => {
      logWarn("cache", "discarding v1 cache");
      logWarn("cloud", "no tool-capable models found");
      const entries = getLogEntries();
      expect(entries).toHaveLength(2);
      expect(entries[0].source).toBe("cache");
      expect(entries[1].source).toBe("cloud");
    });
  });

  describe("logError", () => {
    it("adds an error entry to the buffer", () => {
      logError("local", "API discovery failed");
      const entries = getLogEntries();
      expect(entries).toHaveLength(1);
      expect(entries[0].level).toBe("error");
      expect(entries[0].source).toBe("local");
      expect(entries[0].message).toBe("API discovery failed");
    });

    it("adds entries from different sources", () => {
      logError("cloud", "registration failed");
      logError("settings", "settings write failed");
      logError("stream", "connection reset");
      const entries = getLogEntries();
      expect(entries).toHaveLength(3);
      expect(entries[0].source).toBe("cloud");
      expect(entries[1].source).toBe("settings");
      expect(entries[2].source).toBe("stream");
    });
  });

  // ── ring buffer behavior ──

  describe("ring buffer overflow", () => {
    it("caps entries at MAX_ENTRIES (20)", () => {
      for (let i = 0; i < 25; i++) {
        logError("local", `error ${i}`);
      }
      const entries = getLogEntries();
      expect(entries).toHaveLength(20);
      // Should keep the most recent 20 entries
      expect(entries[0].message).toBe("error 5");
      expect(entries[19].message).toBe("error 24");
    });

    it("preserves insertion order for recent entries", () => {
      logWarn("local", "first");
      logError("cloud", "second");
      logWarn("cache", "third");
      const entries = getLogEntries();
      expect(entries[0].message).toBe("first");
      expect(entries[1].message).toBe("second");
      expect(entries[2].message).toBe("third");
    });
  });

  // ── getRecentEntries ──

  describe("getRecentEntries", () => {
    it("returns the N most recent entries", () => {
      for (let i = 0; i < 10; i++) {
        logWarn("local", `entry ${i}`);
      }
      const recent = getRecentEntries(3);
      expect(recent).toHaveLength(3);
      expect(recent[0].message).toBe("entry 7");
      expect(recent[1].message).toBe("entry 8");
      expect(recent[2].message).toBe("entry 9");
    });

    it("returns all entries if fewer than N", () => {
      logWarn("local", "only one");
      const recent = getRecentEntries(5);
      expect(recent).toHaveLength(1);
      expect(recent[0].message).toBe("only one");
    });

    it("returns empty array if no entries", () => {
      expect(getRecentEntries(5)).toHaveLength(0);
    });

    it("defaults to 5 entries", () => {
      for (let i = 0; i < 10; i++) {
        logError("stream", `err ${i}`);
      }
      const recent = getRecentEntries();
      expect(recent).toHaveLength(5);
      expect(recent[4].message).toBe("err 9");
    });
  });

  // ── getErrors ──

  describe("getErrors", () => {
    it("returns only error-level entries", () => {
      logWarn("local", "just a warning");
      logError("cloud", "actual error");
      logWarn("cache", "another warning");
      logError("settings", "another error");

      const errors = getErrors();
      expect(errors).toHaveLength(2);
      expect(errors[0].level).toBe("error");
      expect(errors[1].level).toBe("error");
    });

    it("returns empty array when there are no errors", () => {
      logWarn("local", "warning only");
      expect(getErrors()).toHaveLength(0);
    });
  });

  // ── clearLog ──

  describe("clearLog", () => {
    it("clears all entries from the buffer", () => {
      logError("local", "error");
      logWarn("cloud", "warning");
      expect(getLogEntries()).toHaveLength(2);

      clearLog();
      expect(getLogEntries()).toHaveLength(0);
    });
  });

  // ── formatStatusLine ──

  describe("formatStatusLine", () => {
    it("returns undefined when there are no entries", () => {
      expect(formatStatusLine()).toBeUndefined();
    });

    it("returns a single error message", () => {
      logError("local", "server unreachable");
      expect(formatStatusLine()).toBe("⚠ ollama: server unreachable");
    });

    it("returns a single warning message", () => {
      logWarn("cache", "discarding v1 cache");
      expect(formatStatusLine()).toBe("⚠ ollama: discarding v1 cache");
    });

    it("summarizes multiple entries", () => {
      logError("local", "server unreachable");
      logError("cloud", "auth failed");
      logError("settings", "write failed");
      const status = formatStatusLine();
      expect(status).toContain("3 errors");
      expect(status).toContain("server unreachable");
      expect(status).toContain("auth failed");
      expect(status).toContain("write failed");
    });

    it("shows overflow count when there are more entries than displayed", () => {
      for (let i = 0; i < 6; i++) {
        logError("local", `error ${i}`);
      }
      // 6 entries, shows 3 recent + overflow
      const status = formatStatusLine();
      expect(status).toContain("+3 more");
    });

    it("does not show overflow when entries fit in display", () => {
      logError("local", "err1");
      logError("local", "err2");
      const status = formatStatusLine();
      expect(status).not.toContain("+");
      expect(status).not.toContain("more");
    });

    it("uses singular 'error' for one error among warnings", () => {
      logError("local", "server unreachable");
      logWarn("cache", "discarding cache");
      const status = formatStatusLine();
      expect(status).toContain("1 error");
      expect(status).toContain("server unreachable");
      expect(status).toContain("discarding cache");
    });

    it("clears status after clearLog", () => {
      logError("local", "some error");
      expect(formatStatusLine()).toBeTruthy();
      clearLog();
      expect(formatStatusLine()).toBeUndefined();
    });
  });

  // ── type checks ──

  describe("LogEntry type", () => {
    it("has all required fields", () => {
      logError("auto-pull", "pull failed");
      const entry = getLogEntries()[0];
      expect(entry).toHaveProperty("timestamp");
      expect(entry).toHaveProperty("level");
      expect(entry).toHaveProperty("source");
      expect(entry).toHaveProperty("message");
      expect(typeof entry.timestamp).toBe("number");
      expect(typeof entry.level).toBe("string");
      expect(typeof entry.source).toBe("string");
      expect(typeof entry.message).toBe("string");
    });
  });
});