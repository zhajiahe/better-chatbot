import { getModelsInfo } from "lib/ai/models";

// Make this route dynamic to fetch fresh models
export const dynamic = "force-dynamic";
export const revalidate = 3600; // Revalidate every hour

// Set to "true" to show all providers including those without API keys
const SHOW_ALL_PROVIDERS = process.env.SHOW_ALL_PROVIDERS === "true";

export const GET = async () => {
  const modelsInfo = await getModelsInfo();

  // Filter to only show providers with configured API keys
  const filteredModels = SHOW_ALL_PROVIDERS
    ? modelsInfo
    : modelsInfo.filter((provider) => provider.hasAPIKey);

  return Response.json(
    filteredModels.sort((a, b) => {
      if (a.hasAPIKey && !b.hasAPIKey) return -1;
      if (!a.hasAPIKey && b.hasAPIKey) return 1;
      return 0;
    }),
  );
};
