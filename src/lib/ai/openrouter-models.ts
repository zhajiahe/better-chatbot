import "server-only";

import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { LanguageModel } from "ai";
import logger from "logger";

// Create OpenRouter provider with API key
const openrouter = createOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY,
});

interface OpenRouterModelInfo {
  id: string;
  name: string;
  pricing: { prompt: string; completion: string };
  architecture: { modality: string };
  supported_parameters: string[];
}

// Simple cache with 1 hour TTL
let cache: {
  models: Record<string, LanguageModel>;
  unsupported: Set<LanguageModel>;
  timestamp: number;
} | null = null;

const CACHE_TTL = 3600_000; // 1 hour

export interface OpenRouterModelsOptions {
  freeOnly?: boolean;
  textOnly?: boolean;
  maxModels?: number;
}

/**
 * Fetch and create OpenRouter models dynamically
 */
export async function createOpenRouterModels(
  options: OpenRouterModelsOptions = {},
): Promise<{
  models: Record<string, LanguageModel>;
  unsupportedModels: Set<LanguageModel>;
}> {
  // Return cached if valid
  if (cache && Date.now() - cache.timestamp < CACHE_TTL) {
    return { models: cache.models, unsupportedModels: cache.unsupported };
  }

  try {
    const response = await fetch("https://openrouter.ai/api/v1/models", {
      headers: { "Content-Type": "application/json" },
      next: { revalidate: 3600 },
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = (await response.json()) as { data: OpenRouterModelInfo[] };
    let items = data.data || [];

    // Apply filters
    if (options.freeOnly) {
      items = items.filter(
        (m) =>
          m.id.includes(":free") ||
          (m.pricing.prompt === "0" && m.pricing.completion === "0"),
      );
    }
    if (options.textOnly) {
      items = items.filter((m) => m.architecture.modality.includes("text"));
    }
    if (options.maxModels) {
      items = items.slice(0, options.maxModels);
    }

    // Build models map
    const models: Record<string, LanguageModel> = {};
    const unsupported = new Set<LanguageModel>();

    for (const info of items) {
      const name = info.id.split("/").slice(1).join("/") || info.id;
      const model = openrouter(info.id);
      models[name] = model;

      if (
        !info.supported_parameters.includes("tools") &&
        !info.supported_parameters.includes("tool_choice")
      ) {
        unsupported.add(model);
      }
    }

    cache = { models, unsupported, timestamp: Date.now() };
    return { models, unsupportedModels: unsupported };
  } catch (error) {
    logger.error("Failed to fetch OpenRouter models:", error);
    return { models: {}, unsupportedModels: new Set() };
  }
}
