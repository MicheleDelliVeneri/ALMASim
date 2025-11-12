import { For, Show } from "solid-js";
import type { JSX } from "solid-js";
import type { MetadataQuery } from "../../lib/api/metadata";

interface MetadataQueryFormProps {
  scienceTypes: { keywords: string[]; categories: string[] } | null;
  loading: boolean;
  onSubmit: (query: MetadataQuery) => void;
}

const bandOptions = [3, 4, 5, 6, 7, 8, 9, 10];

export function MetadataQueryForm(props: MetadataQueryFormProps) {
  const handleSubmit: JSX.EventHandlerUnion<HTMLFormElement, SubmitEvent> = (event) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const query: MetadataQuery = {};

    const keywords = formData.getAll("science_keyword").filter(Boolean) as string[];
    if (keywords.length) query.science_keyword = keywords;

    const categories = formData.getAll("scientific_category").filter(Boolean) as string[];
    if (categories.length) query.scientific_category = categories;

    const bands = formData
      .getAll("bands")
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value));
    if (bands.length) query.bands = bands;

    const fovMin = formData.get("fov_min");
    const fovMax = formData.get("fov_max");
    if (fovMin && fovMax) {
      const min = Number(fovMin);
      const max = Number(fovMax);
      if (Number.isFinite(min) && Number.isFinite(max)) {
        query.fov_range = [min, max];
      }
    }

    const timeMin = formData.get("time_min");
    const timeMax = formData.get("time_max");
    if (timeMin && timeMax) {
      const min = Number(timeMin);
      const max = Number(timeMax);
      if (Number.isFinite(min) && Number.isFinite(max)) {
        query.time_resolution_range = [min, max];
      }
    }

    const freqMin = formData.get("freq_min");
    const freqMax = formData.get("freq_max");
    if (freqMin && freqMax) {
      const min = Number(freqMin);
      const max = Number(freqMax);
      if (Number.isFinite(min) && Number.isFinite(max)) {
        query.frequency_range = [min, max];
      }
    }

    props.onSubmit(query);
  };

  return (
    <form onSubmit={handleSubmit} class="bg-white rounded-lg shadow p-6 space-y-4">
      <div class="flex items-center justify-between">
        <h2 class="text-xl font-semibold text-gray-900">Query Builder</h2>
        <button
          type="submit"
          disabled={props.loading}
          class="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-400"
        >
          {props.loading ? "Running..." : "Run Query"}
        </button>
      </div>

      <div class="grid grid-cols-1 gap-4 md:grid-cols-2">
        <label class="block">
          <span class="text-sm font-medium text-gray-700">Science Keywords</span>
          <select
            name="science_keyword"
            multiple
            class="mt-1 w-full border border-gray-300 rounded-md p-2 h-32"
          >
            <Show
              when={props.scienceTypes}
              fallback={<option>Loading keywords...</option>}
            >
              <For each={props.scienceTypes?.keywords || []}>
                {(keyword) => <option value={keyword}>{keyword}</option>}
              </For>
            </Show>
          </select>
        </label>

        <label class="block">
          <span class="text-sm font-medium text-gray-700">Scientific Categories</span>
          <select
            name="scientific_category"
            multiple
            class="mt-1 w-full border border-gray-300 rounded-md p-2 h-32"
          >
            <Show
              when={props.scienceTypes}
              fallback={<option>Loading categories...</option>}
            >
              <For each={props.scienceTypes?.categories || []}>
                {(category) => <option value={category}>{category}</option>}
              </For>
            </Show>
          </select>
        </label>
      </div>

      <div>
        <span class="text-sm font-medium text-gray-700">Bands</span>
        <div class="mt-2 flex flex-wrap gap-2">
          <For each={bandOptions}>
            {(band) => (
              <label class="inline-flex items-center space-x-1 text-sm text-gray-700">
                <input type="checkbox" name="bands" value={band} class="rounded border-gray-300" />
                <span>Band {band}</span>
              </label>
            )}
          </For>
        </div>
      </div>

      <div class="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div>
          <span class="block text-sm font-medium text-gray-700">FOV (arcsec)</span>
          <div class="mt-1 flex gap-2">
            <input
              type="number"
              step="0.1"
              name="fov_min"
              placeholder="Min"
              class="w-full rounded-md border border-gray-300 p-2"
            />
            <input
              type="number"
              step="0.1"
              name="fov_max"
              placeholder="Max"
              class="w-full rounded-md border border-gray-300 p-2"
            />
          </div>
        </div>

        <div>
          <span class="block text-sm font-medium text-gray-700">Time Resolution (s)</span>
          <div class="mt-1 flex gap-2">
            <input
              type="number"
              step="0.1"
              name="time_min"
              placeholder="Min"
              class="w-full rounded-md border border-gray-300 p-2"
            />
            <input
              type="number"
              step="0.1"
              name="time_max"
              placeholder="Max"
              class="w-full rounded-md border border-gray-300 p-2"
            />
          </div>
        </div>

        <div>
          <span class="block text-sm font-medium text-gray-700">Frequency (GHz)</span>
          <div class="mt-1 flex gap-2">
            <input
              type="number"
              step="0.1"
              name="freq_min"
              placeholder="Min"
              class="w-full rounded-md border border-gray-300 p-2"
            />
            <input
              type="number"
              step="0.1"
              name="freq_max"
              placeholder="Max"
              class="w-full rounded-md border border-gray-300 p-2"
            />
          </div>
        </div>
      </div>
    </form>
  );
}

