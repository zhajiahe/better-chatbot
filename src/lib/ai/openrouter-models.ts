import "server-only";

import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { LanguageModel } from "ai";
import logger from "logger";

// Create OpenRouter provider with API key from environment
const openrouter = createOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY,
});

interface OpenRouterModelInfo {
  id: string;
  name: string;
  context_length: number;
  architecture: {
    modality: string;
    input_modalities: string[];
    output_modalities: string[];
  };
  pricing: {
    prompt: string;
    completion: string;
  };
  supported_parameters: string[];
}

interface OpenRouterModelsResponse {
  data: OpenRouterModelInfo[];
}

// Cache for OpenRouter models
let cachedModels: Record<string, LanguageModel> | null = null;
let cachedModelInfos: OpenRouterModelInfo[] | null = null;
let cacheTimestamp: number = 0;
const CACHE_DURATION_MS = 1000 * 60 * 60; // 1 hour

/**
 * Fetch all available models from OpenRouter API
 */
async function fetchOpenRouterModels(): Promise<OpenRouterModelInfo[]> {
  try {
    const response = await fetch("https://openrouter.ai/api/v1/models", {
      headers: {
        "Content-Type": "application/json",
      },
      next: { revalidate: 3600 }, // Cache for 1 hour in Next.js
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch OpenRouter models: ${response.status}`);
    }

    const data: OpenRouterModelsResponse = await response.json();
    return data.data || [];
  } catch (error) {
    logger.error("Failed to fetch OpenRouter models:", error);
    return [];
  }
}

/**
 * Filter models based on configuration
 */
function filterModels(
  models: OpenRouterModelInfo[],
  options: OpenRouterModelsOptions,
): OpenRouterModelInfo[] {
  let filtered = models;

  // Filter by free models only
  if (options.freeOnly) {
    filtered = filtered.filter(
      (m) =>
        m.id.includes(":free") ||
        (m.pricing.prompt === "0" && m.pricing.completion === "0"),
    );
  }

  // Filter by text modality
  if (options.textOnly) {
    filtered = filtered.filter(
      (m) =>
        m.architecture.modality === "text->text" ||
        m.architecture.modality.includes("text"),
    );
  }

  // Filter by supported parameters (e.g., tools)
  if (options.requireTools) {
    filtered = filtered.filter(
      (m) =>
        m.supported_parameters.includes("tools") ||
        m.supported_parameters.includes("tool_choice"),
    );
  }

  // Apply max models limit
  if (options.maxModels && options.maxModels > 0) {
    filtered = filtered.slice(0, options.maxModels);
  }

  return filtered;
}

export interface OpenRouterModelsOptions {
  /** Only include free models */
  freeOnly?: boolean;
  /** Only include text-to-text models */
  textOnly?: boolean;
  /** Only include models that support tool calling */
  requireTools?: boolean;
  /** Maximum number of models to include */
  maxModels?: number;
  /** Custom filter function */
  customFilter?: (model: OpenRouterModelInfo) => boolean;
}

/**
 * Create OpenRouter models dynamically from the API
 */
export async function createOpenRouterModels(
  options: OpenRouterModelsOptions = {},
): Promise<{
  models: Record<string, LanguageModel>;
  modelInfos: OpenRouterModelInfo[];
  unsupportedModels: Set<LanguageModel>;
}> {
  const now = Date.now();

  // Return cached models if still valid
  if (
    cachedModels &&
    cachedModelInfos &&
    now - cacheTimestamp < CACHE_DURATION_MS
  ) {
    const unsupportedModels = new Set<LanguageModel>();
    const filteredInfos = filterModels(cachedModelInfos, options);

    // Create models from cached info
    const models: Record<string, LanguageModel> = {};
    for (const info of filteredInfos) {
      const displayName = getDisplayName(info);
      const model = openrouter(info.id);
      models[displayName] = model;

      // Mark as unsupported if no tool support
      if (
        !info.supported_parameters.includes("tools") &&
        !info.supported_parameters.includes("tool_choice")
      ) {
        unsupportedModels.add(model);
      }
    }

    return { models, modelInfos: filteredInfos, unsupportedModels };
  }

  // Fetch fresh models
  const allModels = await fetchOpenRouterModels();
  cachedModelInfos = allModels;
  cacheTimestamp = now;

  // Apply filters
  const filteredModels = filterModels(allModels, options);

  // Apply custom filter if provided
  const finalModels = options.customFilter
    ? filteredModels.filter(options.customFilter)
    : filteredModels;

  // Create model instances
  const models: Record<string, LanguageModel> = {};
  const unsupportedModels = new Set<LanguageModel>();

  for (const info of finalModels) {
    const displayName = getDisplayName(info);
    const model = openrouter(info.id);
    models[displayName] = model;

    // Mark as unsupported if no tool support
    if (
      !info.supported_parameters.includes("tools") &&
      !info.supported_parameters.includes("tool_choice")
    ) {
      unsupportedModels.add(model);
    }
  }

  cachedModels = models;

  return { models, modelInfos: finalModels, unsupportedModels };
}

/**
 * Get a human-readable display name for the model
 */
function getDisplayName(info: OpenRouterModelInfo): string {
  // Use the model ID but make it more readable
  // e.g., "openai/gpt-4o" -> "gpt-4o"
  // e.g., "anthropic/claude-3.5-sonnet" -> "claude-3.5-sonnet"
  const parts = info.id.split("/");
  return parts.length > 1 ? parts.slice(1).join("/") : info.id;
}

/**
 * Check if a model supports image input
 */
export function isOpenRouterModelImageSupported(
  modelInfos: OpenRouterModelInfo[],
  modelId: string,
): boolean {
  const info = modelInfos.find((m) => m.id === modelId);
  if (!info) return false;

  return (
    info.architecture.input_modalities.includes("image") ||
    info.architecture.modality.includes("image")
  );
}

/**
 * Get OpenRouter models synchronously (uses cache or returns empty)
 * This is useful for initial server render where async isn't ideal
 */
export function getOpenRouterModelsSync(): Record<string, LanguageModel> {
  return cachedModels || {};
}

/**
 * Get cached model infos
 */
export function getOpenRouterModelInfosSync(): OpenRouterModelInfo[] {
  return cachedModelInfos || [];
}

/**
 * Initialize the cache (call this on server startup)
 */
export async function initOpenRouterModelsCache(
  options: OpenRouterModelsOptions = {},
): Promise<void> {
  await createOpenRouterModels(options);
}
