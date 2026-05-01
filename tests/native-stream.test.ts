/**
 * Tests for native-stream.ts — NDJSON parsing, message conversion,
 * tool conversion, ghost-token detection, overflow detection.
 */

import { describe, it, expect } from "vitest";
import { Readable } from "node:stream";

import {
  parseNDJSON,
  convertMessages,
  convertTools,
  isGhostTokenStream,
  isOllamaContextOverflow,
  type OllamaChatChunk,
  type OllamaToolCall,
} from "../extensions/pi-ollama-provider/native-stream.js";

// ════════════════════════════════════════════════════════════════
// parseNDJSON
// ════════════════════════════════════════════════════════════════

describe("parseNDJSON", () => {
  it("parses single-line NDJSON from Node.js Readable", async () => {
    const data = JSON.stringify({ message: { content: "hello" }, done: false }) + "\n";
    const stream = Readable.from([Buffer.from(data)]);
    const chunks: OllamaChatChunk[] = [];
    for await (const chunk of parseNDJSON(stream as any)) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(1);
    expect(chunks[0].message?.content).toBe("hello");
  });

  it("parses multi-line NDJSON", async () => {
    const lines = [
      JSON.stringify({ message: { content: "hel" }, done: false }),
      JSON.stringify({ message: { content: "lo" }, done: false }),
      JSON.stringify({ message: { content: "!" }, done: true, eval_count: 3 }),
    ].join("\n") + "\n";
    const stream = Readable.from([Buffer.from(lines)]);
    const chunks: OllamaChatChunk[] = [];
    for await (const chunk of parseNDJSON(stream as any)) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(3);
    expect(chunks[2].done).toBe(true);
    expect(chunks[2].eval_count).toBe(3);
  });

  it("handles streaming chunks (partial lines)", async () => {
    // Simulates data arriving in chunks that split NDJSON lines
    const full = JSON.stringify({ message: { content: "test" }, done: false }) + "\n" +
      JSON.stringify({ message: { content: "more" }, done: true, eval_count: 2 }) + "\n";
    const stream = Readable.from([
      Buffer.from(full.slice(0, 20)),
      Buffer.from(full.slice(20)),
    ]);
    const chunks: OllamaChatChunk[] = [];
    for await (const chunk of parseNDJSON(stream as any)) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(2);
  });

  it("skips malformed lines", async () => {
    const data = 'not-json\n{"message":{"content":"ok"},"done":true}\n\n';
    const stream = Readable.from([Buffer.from(data)]);
    const chunks: OllamaChatChunk[] = [];
    for await (const chunk of parseNDJSON(stream as any)) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(1);
    expect(chunks[0].done).toBe(true);
  });

  it("handles empty stream", async () => {
    const stream = Readable.from([]);
    const chunks: OllamaChatChunk[] = [];
    for await (const chunk of parseNDJSON(stream as any)) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(0);
  });

  it("handles null body", async () => {
    const chunks: OllamaChatChunk[] = [];
    for await (const chunk of parseNDJSON(null)) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(0);
  });
});

// ════════════════════════════════════════════════════════════════
// convertMessages
// ════════════════════════════════════════════════════════════════

describe("convertMessages", () => {
  it("converts developer role to system", () => {
    const messages = [{ role: "developer", content: "You are helpful." }];
    const result = convertMessages(messages);
    expect(result[0].role).toBe("system");
    expect(result[0].content).toBe("You are helpful.");
  });

  it("preserves system and user roles", () => {
    const messages = [
      { role: "system", content: "System prompt" },
      { role: "user", content: "User message" },
    ];
    const result = convertMessages(messages);
    expect(result[0].role).toBe("system");
    expect(result[1].role).toBe("user");
  });

  it("handles multipart content with images (vision)", () => {
    const messages = [
      {
        role: "user",
        content: [
          { type: "text", text: "What's in this image?" },
          {
            type: "image_url",
            image_url: { url: "data:image/png;base64,iVBOR..." },
          },
        ],
      },
    ];
    const result = convertMessages(messages, true);
    expect(result[0].content).toBe("What's in this image?");
    expect(result[0].images).toHaveLength(1);
    expect(result[0].images![0]).toBe("iVBOR...");
  });

  it("strips images when model doesn't support vision", () => {
    const messages = [
      {
        role: "user",
        content: [
          { type: "text", text: "What's in this image?" },
          {
            type: "image_url",
            image_url: { url: "data:image/png;base64,iVBOR..." },
          },
        ],
      },
    ];
    const result = convertMessages(messages, false);
    expect(result[0].content).toBe("What's in this image?");
    expect(result[0].images).toBeUndefined();
  });

  it("converts tool result messages", () => {
    const messages = [
      {
        role: "tool",
        tool_call_id: "call_123",
        content: '{"result": "success"}',
      },
    ];
    const result = convertMessages(messages);
    expect(result[0].role).toBe("tool");
    expect(result[0].tool_call_id).toBe("call_123");
  });

  it("preserves assistant tool_calls", () => {
    const toolCalls: OllamaToolCall[] = [
      { function: { name: "read_file", arguments: { path: "/foo" } } },
    ];
    const messages = [
      {
        role: "assistant",
        content: null,
        tool_calls: toolCalls,
      },
    ];
    const result = convertMessages(messages);
    expect(result[0].tool_calls).toHaveLength(1);
    expect(result[0].tool_calls![0].function.name).toBe("read_file");
  });
});

// ════════════════════════════════════════════════════════════════
// convertTools
// ════════════════════════════════════════════════════════════════

describe("convertTools", () => {
  it("converts function tools to Ollama format", () => {
    const tools = [
      {
        type: "function",
        function: {
          name: "read_file",
          description: "Read a file",
          parameters: { type: "object", properties: { path: { type: "string" } } },
        },
      },
    ];
    const result = convertTools(tools);
    expect(result).toHaveLength(1);
    expect(result[0].type).toBe("function");
    expect(result[0].function.name).toBe("read_file");
  });

  it("returns empty array for undefined", () => {
    expect(convertTools(undefined)).toEqual([]);
  });

  it("returns empty array for empty array", () => {
    expect(convertTools([])).toEqual([]);
  });

  it("filters out non-function tools", () => {
    const tools = [
      { type: "function", function: { name: "test", parameters: {} } },
      { type: "other", function: { name: "bad", parameters: {} } },
    ];
    expect(convertTools(tools as any)).toHaveLength(1);
  });
});

// ════════════════════════════════════════════════════════════════
// isGhostTokenStream
// ════════════════════════════════════════════════════════════════

describe("isGhostTokenStream", () => {
  it("detects ghost token: done=true, eval_count>0, no content or tool_calls", () => {
    const chunks: OllamaChatChunk[] = [
      { model: "test", created_at: "", message: { role: "assistant", content: "" }, done: false },
      { model: "test", created_at: "", done: true, eval_count: 42, prompt_eval_count: 10 },
    ];
    expect(isGhostTokenStream(chunks)).toBe(true);
  });

  it("not ghost: has content", () => {
    const chunks: OllamaChatChunk[] = [
      { model: "test", created_at: "", message: { role: "assistant", content: "Hello" }, done: false },
      { model: "test", created_at: "", done: true, eval_count: 1 },
    ];
    expect(isGhostTokenStream(chunks)).toBe(false);
  });

  it("not ghost: has tool calls", () => {
    const chunks: OllamaChatChunk[] = [
      {
        model: "test", created_at: "",
        message: { role: "assistant", content: "", tool_calls: [{ function: { name: "test", arguments: {} } }] },
        done: false,
      },
      { model: "test", created_at: "", done: true, eval_count: 5 },
    ];
    expect(isGhostTokenStream(chunks)).toBe(false);
  });

  it("not ghost: eval_count is 0", () => {
    const chunks: OllamaChatChunk[] = [
      { model: "test", created_at: "", message: { role: "assistant", content: "" }, done: false },
      { model: "test", created_at: "", done: true, eval_count: 0 },
    ];
    expect(isGhostTokenStream(chunks)).toBe(false);
  });

  it("not ghost: stream not finished (no done:true)", () => {
    const chunks: OllamaChatChunk[] = [
      { model: "test", created_at: "", message: { role: "assistant", content: "" }, done: false },
    ];
    expect(isGhostTokenStream(chunks)).toBe(false);
  });

  it("not ghost: empty chunks array", () => {
    expect(isGhostTokenStream([])).toBe(false);
  });
});

// ════════════════════════════════════════════════════════════════
// isOllamaContextOverflow
// ════════════════════════════════════════════════════════════════

describe("isOllamaContextOverflow", () => {
  it("detects 'exceeded max context length'", () => {
    expect(isOllamaContextOverflow("prompt too long; exceeded max context length by 1234 tokens")).toBe(true);
  });

  it("detects 'prompt too long'", () => {
    expect(isOllamaContextOverflow("prompt too long for model context")).toBe(true);
  });

  it("detects 'context window exceeded'", () => {
    expect(isOllamaContextOverflow("context window exceeded")).toBe(true);
  });

  it("detects 'maximum context length exceeded'", () => {
    expect(isOllamaContextOverflow("maximum context length exceeded")).toBe(true);
  });

  it("no match for unrelated errors", () => {
    expect(isOllamaContextOverflow("model not found")).toBe(false);
    expect(isOllamaContextOverflow("connection refused")).toBe(false);
  });

  it("handles Error objects", () => {
    expect(isOllamaContextOverflow(new Error("exceeded max context length"))).toBe(true);
  });

  it("handles null/undefined", () => {
    expect(isOllamaContextOverflow(null)).toBe(false);
    expect(isOllamaContextOverflow(undefined)).toBe(false);
  });
});