import { describe, it, expect } from "vitest";

// Fixed cloud detection logic (matches index.ts)
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

// Generate model ID (matches index.ts)
function generateModelId(
  modelName: string,
  isCloud: boolean,
): string {
  if (isCloud && !modelName.endsWith("-cloud") && !modelName.includes(":cloud")) {
    return modelName.includes(":") ? `${modelName}-cloud` : `${modelName}:cloud`;
  }
  return modelName;
}

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

      // After ollama pull X:cloud, the model ID already has the suffix
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