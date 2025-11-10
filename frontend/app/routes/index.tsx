import { createResource } from "solid-js";
import { apiClient } from "../lib/api/client";

export default function Home() {
  const [healthStatus] = createResource(async () => {
    try {
      const response = await apiClient.get<{ status: string }>("/health");
      return response.status === "healthy" ? "✅ Healthy" : "❌ Unhealthy";
    } catch (error) {
      return "❌ Error connecting to API";
    }
  });

  return (
    <div class="container mx-auto px-4 py-8">
      <div class="max-w-4xl mx-auto">
        <header class="mb-8">
          <h1 class="text-4xl font-bold text-gray-900 mb-2">ALMASim</h1>
          <p class="text-lg text-gray-600">
            Generate realistic ALMA observations with advanced simulation capabilities
          </p>
          <div class="mt-4">
            <span class="inline-block px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
              API Status: {healthStatus() || "checking..."}
            </span>
          </div>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
          <a
            href="/simulations"
            class="block p-6 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow"
          >
            <h2 class="text-xl font-semibold text-gray-900 mb-2">Simulations</h2>
            <p class="text-gray-600">Create and manage ALMA simulations</p>
          </a>

          <a
            href="/metadata"
            class="block p-6 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow"
          >
            <h2 class="text-xl font-semibold text-gray-900 mb-2">Metadata</h2>
            <p class="text-gray-600">Query and browse ALMA observation metadata</p>
          </a>
        </div>
      </div>
    </div>
  );
}

