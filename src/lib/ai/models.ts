import "server-only";

import { createOllama } from "ollama-ai-provider-v2";
import { openai } from "@ai-sdk/openai";
import { google } from "@ai-sdk/google";
import { anthropic } from "@ai-sdk/anthropic";
import { xai } from "@ai-sdk/xai";
import { LanguageModelV2 } from "@openrouter/ai-sdk-provider";
import { createGroq } from "@ai-sdk/groq";
import { LanguageModel } from "ai";
import {
  createOpenAICompatibleModels,
  openaiCompatibleModelsSafeParse,
} from "./create-openai-compatiable";
import { ChatModel } from "app-types/chat";
import {
  DEFAULT_FILE_PART_MIME_TYPES,
  OPENAI_FILE_MIME_TYPES,
  GEMINI_FILE_MIME_TYPES,
  ANTHROPIC_FILE_MIME_TYPES,
  XAI_FILE_MIME_TYPES,
} from "./file-support";
import {
  createOpenRouterModels,
  OpenRouterModelsOptions,
} from "./openrouter-models";

const ollama = createOllama({
  baseURL: process.env.OLLAMA_BASE_URL || "http://localhost:11434/api",
});
const groq = createGroq({
  baseURL: process.env.GROQ_BASE_URL || "https://api.groq.com/openai/v1",
  apiKey: process.env.GROQ_API_KEY,
});

const staticModels = {
  openai: {
    "gpt-4.1": openai("gpt-4.1"),
    "gpt-4.1-mini": openai("gpt-4.1-mini"),
    "o4-mini": openai("o4-mini"),
    o3: openai("o3"),
    "gpt-5.1-chat": openai("gpt-5.1-chat-latest"),
    "gpt-5.1": openai("gpt-5.1"),
    "gpt-5.1-codex": openai("gpt-5.1-codex"),
    "gpt-5.1-codex-mini": openai("gpt-5.1-codex-mini"),
  },
  google: {
    "gemini-2.5-flash-lite": google("gemini-2.5-flash-lite"),
    "gemini-2.5-flash": google("gemini-2.5-flash"),
    "gemini-3-pro": google("gemini-3-pro-preview"),
    "gemini-2.5-pro": google("gemini-2.5-pro"),
  },
  anthropic: {
    "sonnet-4.5": anthropic("claude-sonnet-4-5"),
    "haiku-4.5": anthropic("claude-haiku-4-5"),
    "opus-4.5": anthropic("claude-opus-4-5"),
  },
  xai: {
    "grok-4-1-fast": xai("grok-4-1-fast-non-reasoning"),
    "grok-4-1": xai("grok-4-1"),
    "grok-3-mini": xai("grok-3-mini"),
  },
  ollama: {
    "gemma3:1b": ollama("gemma3:1b"),
    "gemma3:4b": ollama("gemma3:4b"),
    "gemma3:12b": ollama("gemma3:12b"),
  },
  groq: {
    "kimi-k2-instruct": groq("moonshotai/kimi-k2-instruct"),
    "llama-4-scout-17b": groq("meta-llama/llama-4-scout-17b-16e-instruct"),
    "gpt-oss-20b": groq("openai/gpt-oss-20b"),
    "gpt-oss-120b": groq("openai/gpt-oss-120b"),
    "qwen3-32b": groq("qwen/qwen3-32b"),
  },
  // OpenRouter models are loaded dynamically via createOpenRouterModels()
  // Keep minimal static models as fallback
  openRouter: {} as Record<string, LanguageModel>,
};

const staticUnsupportedModels = new Set([
  staticModels.openai["o4-mini"],
  staticModels.ollama["gemma3:1b"],
  staticModels.ollama["gemma3:4b"],
  staticModels.ollama["gemma3:12b"],
]);

const staticSupportImageInputModels = {
  ...staticModels.google,
  ...staticModels.xai,
  ...staticModels.openai,
  ...staticModels.anthropic,
};

const staticFilePartSupportByModel = new Map<
  LanguageModel,
  readonly string[]
>();

const registerFileSupport = (
  model: LanguageModel | undefined,
  mimeTypes: readonly string[] = DEFAULT_FILE_PART_MIME_TYPES,
) => {
  if (!model) return;
  staticFilePartSupportByModel.set(model, Array.from(mimeTypes));
};

registerFileSupport(staticModels.openai["gpt-4.1"], OPENAI_FILE_MIME_TYPES);
registerFileSupport(
  staticModels.openai["gpt-4.1-mini"],
  OPENAI_FILE_MIME_TYPES,
);
registerFileSupport(staticModels.openai["gpt-5"], OPENAI_FILE_MIME_TYPES);
registerFileSupport(staticModels.openai["gpt-5-mini"], OPENAI_FILE_MIME_TYPES);
registerFileSupport(staticModels.openai["gpt-5-nano"], OPENAI_FILE_MIME_TYPES);

registerFileSupport(
  staticModels.google["gemini-2.5-flash-lite"],
  GEMINI_FILE_MIME_TYPES,
);
registerFileSupport(
  staticModels.google["gemini-2.5-flash"],
  GEMINI_FILE_MIME_TYPES,
);
registerFileSupport(
  staticModels.google["gemini-2.5-pro"],
  GEMINI_FILE_MIME_TYPES,
);

registerFileSupport(
  staticModels.anthropic["sonnet-4.5"],
  ANTHROPIC_FILE_MIME_TYPES,
);
registerFileSupport(
  staticModels.anthropic["opus-4.1"],
  ANTHROPIC_FILE_MIME_TYPES,
);

registerFileSupport(staticModels.xai["grok-4-fast"], XAI_FILE_MIME_TYPES);
registerFileSupport(staticModels.xai["grok-4"], XAI_FILE_MIME_TYPES);
registerFileSupport(staticModels.xai["grok-3"], XAI_FILE_MIME_TYPES);
registerFileSupport(staticModels.xai["grok-3-mini"], XAI_FILE_MIME_TYPES);
// OpenRouter file support is determined dynamically based on model capabilities

const openaiCompatibleProviders = openaiCompatibleModelsSafeParse(
  process.env.OPENAI_COMPATIBLE_DATA,
);

const {
  providers: openaiCompatibleModels,
  unsupportedModels: openaiCompatibleUnsupportedModels,
} = createOpenAICompatibleModels(openaiCompatibleProviders);

// Base static models (excluding dynamic openRouter)
const baseStaticModels = { ...openaiCompatibleModels, ...staticModels };

const baseUnsupportedModels = new Set([
  ...openaiCompatibleUnsupportedModels,
  ...staticUnsupportedModels,
]);

// Dynamic unsupported models cache (for OpenRouter)
let dynamicUnsupportedModels = new Set<LanguageModel>();

export const isToolCallUnsupportedModel = (model: LanguageModel) => {
  return (
    baseUnsupportedModels.has(model) || dynamicUnsupportedModels.has(model)
  );
};

const isImageInputUnsupportedModel = (model: LanguageModel) => {
  return !Object.values(staticSupportImageInputModels).includes(
    model as LanguageModelV2,
  );
};

export const getFilePartSupportedMimeTypes = (model: LanguageModel) => {
  return staticFilePartSupportByModel.get(model) ?? [];
};

// Dynamic fallback model - will be set based on available API keys
function getFallbackModel(): LanguageModel {
  // Check which API keys are configured and return appropriate fallback
  if (process.env.OPENAI_API_KEY && process.env.OPENAI_API_KEY !== "****") {
    return staticModels.openai["gpt-4.1"];
  }
  if (
    process.env.GOOGLE_GENERATIVE_AI_API_KEY &&
    process.env.GOOGLE_GENERATIVE_AI_API_KEY !== "****"
  ) {
    return staticModels.google["gemini-2.5-flash"];
  }
  if (
    process.env.ANTHROPIC_API_KEY &&
    process.env.ANTHROPIC_API_KEY !== "****"
  ) {
    return staticModels.anthropic["sonnet-4.5"];
  }
  if (process.env.XAI_API_KEY && process.env.XAI_API_KEY !== "****") {
    return staticModels.xai["grok-3-mini"];
  }
  if (process.env.GROQ_API_KEY && process.env.GROQ_API_KEY !== "****") {
    return staticModels.groq["qwen3-32b"];
  }
  // OpenRouter fallback - will be set dynamically after models are loaded
  // For now, return a placeholder that will be updated
  return staticModels.openai["gpt-4.1"]; // This should be overridden
}

// Cache for dynamic models
let cachedAllModels: Record<
  string,
  Record<string, LanguageModel>
> = baseStaticModels;

/**
 * OpenRouter model loading options
 */
export const openRouterOptions: OpenRouterModelsOptions = {
  // Default: load all text models (can be customized via env)
  textOnly: process.env.OPENROUTER_TEXT_ONLY !== "false",
  freeOnly: process.env.OPENROUTER_FREE_ONLY === "true",
  maxModels: process.env.OPENROUTER_MAX_MODELS
    ? parseInt(process.env.OPENROUTER_MAX_MODELS, 10)
    : undefined,
};

/**
 * Get models info with dynamic OpenRouter models
 * This is async and should be called from API routes
 */
export async function getModelsInfo() {
  const hasOpenRouterKey = checkProviderAPIKey("openRouter");

  // Start with static models info
  const staticModelsInfo = Object.entries(baseStaticModels)
    .filter(([provider]) => provider !== "openRouter") // Exclude static openRouter placeholder
    .map(([provider, models]) => ({
      provider,
      models: Object.entries(models).map(([name, model]) => ({
        name,
        isToolCallUnsupported: isToolCallUnsupportedModel(model),
        isImageInputUnsupported: isImageInputUnsupportedModel(model),
        supportedFileMimeTypes: [...getFilePartSupportedMimeTypes(model)],
      })),
      hasAPIKey: checkProviderAPIKey(provider as keyof typeof staticModels),
    }));

  // If OpenRouter API key is configured, fetch dynamic models
  if (hasOpenRouterKey) {
    try {
      const { models, unsupportedModels } =
        await createOpenRouterModels(openRouterOptions);

      // Update dynamic unsupported models cache
      dynamicUnsupportedModels = unsupportedModels;

      // Update cached models for getModel()
      cachedAllModels = {
        ...baseStaticModels,
        openRouter: models,
      };

      // Add OpenRouter models info
      const openRouterModelsInfo = {
        provider: "openRouter",
        models: Object.entries(models).map(([name, model]) => ({
          name,
          isToolCallUnsupported: unsupportedModels.has(model),
          isImageInputUnsupported: true, // Most OpenRouter models don't support image input
          supportedFileMimeTypes: [] as string[],
        })),
        hasAPIKey: true,
      };

      return [...staticModelsInfo, openRouterModelsInfo];
    } catch (error) {
      console.error("Failed to load OpenRouter models:", error);
      // Fall through to return static models only
    }
  }

  // Return static models info (OpenRouter without models if no key)
  return [
    ...staticModelsInfo,
    {
      provider: "openRouter",
      models: [],
      hasAPIKey: hasOpenRouterKey,
    },
  ];
}

/**
 * Synchronous model provider for use in chat routes
 * Uses cached models from last getModelsInfo() call
 */
export const customModelProvider = {
  // This is now a getter that returns cached info for backwards compatibility
  // For fresh data, use getModelsInfo() async function
  get modelsInfo() {
    return Object.entries(cachedAllModels)
      .filter(
        ([provider]) =>
          provider !== "openRouter" ||
          Object.keys(cachedAllModels.openRouter || {}).length > 0,
      )
      .map(([provider, models]) => ({
        provider,
        models: Object.entries(models).map(([name, model]) => ({
          name,
          isToolCallUnsupported: isToolCallUnsupportedModel(model),
          isImageInputUnsupported: isImageInputUnsupportedModel(model),
          supportedFileMimeTypes: [...getFilePartSupportedMimeTypes(model)],
        })),
        hasAPIKey: checkProviderAPIKey(provider as keyof typeof staticModels),
      }));
  },
  getModel: (model?: ChatModel): LanguageModel => {
    if (!model) return getFallbackModel();
    const foundModel = cachedAllModels[model.provider]?.[model.model];
    if (!foundModel) {
      // If the requested model is not found, throw an error instead of silently falling back
      // This helps users understand that their model selection is invalid
      throw new Error(
        `Model "${model.model}" from provider "${model.provider}" not found. ` +
          `This may be because the model cache hasn't been initialized yet. ` +
          `Please refresh the page or ensure the API key is configured.`,
      );
    }
    return foundModel;
  },
  /**
   * Async version of getModel that ensures OpenRouter models are loaded first
   */
  async getModelAsync(model?: ChatModel): Promise<LanguageModel> {
    if (!model) return getFallbackModel();

    // If requesting OpenRouter model and cache is empty, fetch models first
    if (
      model.provider === "openRouter" &&
      Object.keys(cachedAllModels.openRouter || {}).length === 0
    ) {
      const hasKey = checkProviderAPIKey("openRouter");
      if (hasKey) {
        await getModelsInfo(); // This will populate the cache
      }
    }

    const foundModel = cachedAllModels[model.provider]?.[model.model];
    if (!foundModel) {
      throw new Error(
        `Model "${model.model}" from provider "${model.provider}" not found. ` +
          `Please select a valid model or configure the appropriate API key.`,
      );
    }
    return foundModel;
  },
};

function checkProviderAPIKey(provider: keyof typeof staticModels) {
  let key: string | undefined;
  switch (provider) {
    case "openai":
      key = process.env.OPENAI_API_KEY;
      break;
    case "google":
      key = process.env.GOOGLE_GENERATIVE_AI_API_KEY;
      break;
    case "anthropic":
      key = process.env.ANTHROPIC_API_KEY;
      break;
    case "xai":
      key = process.env.XAI_API_KEY;
      break;
    case "groq":
      key = process.env.GROQ_API_KEY;
      break;
    case "openRouter":
      key = process.env.OPENROUTER_API_KEY;
      break;
    default:
      return true; // assume the provider has an API key
  }
  return !!key && key != "****";
}
