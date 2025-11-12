import { For, Show, createMemo } from "solid-js";
import type { MetadataResponse } from "../../lib/api/metadata";
import { SkeletonCell } from "../shared/Loader";

interface MetadataResultsTableProps {
  results: MetadataResponse | null;
  loading: boolean;
  onClear: () => void;
  onLoad: () => void;
  onSave: () => void;
  saving: boolean;
}

const defaultColumns = [
  "source_name",
  "project_name",
  "band",
  "ra",
  "dec",
  "fov",
  "time_resolution",
];

const columnLabels: Record<string, string> = {
  source_name: "Source",
  project_name: "Project",
  band: "Band",
  ra: "R.A. (deg)",
  dec: "Dec (deg)",
  fov: "FOV (arcsec)",
  time_resolution: "Time Res. (s)",
  freq_support: "Freq. Support",
  antenna_array: "Antenna Array",
  freq: "Frequency (GHz)",
  ang_res: "Ang. Res. (arcsec)",
  vel_res: "Vel. Res. (km/s)",
  cont_sens: "Continuum Sens.",
  source_type: "Source Type",
  obs_date: "Obs. Date",
};

const placeholderRows = Array.from({ length: 5 }, (_, index) => index);

const formatColumnName = (column: string) => {
  if (columnLabels[column]) return columnLabels[column];
  return column
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

const stringifyValue = (value: unknown) => {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object" && value) return JSON.stringify(value);
  return value?.toString() ?? "";
};

export function MetadataResultsTable(props: MetadataResultsTableProps) {
  const resultColumns = () => {
    const data = props.results?.data;
    if (!data || data.length === 0) return [] as string[];
    return Object.keys(data[0]);
  };

  const tableColumns = createMemo(() => {
    const dynamic = resultColumns();
    return dynamic.length ? dynamic : defaultColumns;
  });

  return (
    <section class="bg-white rounded-lg shadow p-6">
      <div class="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 class="text-xl font-semibold text-gray-900">Results</h2>
          <span class="text-sm text-gray-600">
            {props.results?.count ? `${props.results?.count} matches` : "No data yet"}
          </span>
        </div>
        <div class="flex flex-wrap gap-2">
          <button
            type="button"
            class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed"
            onClick={props.onClear}
            disabled={props.loading || !props.results}
          >
            Clear Metadata
          </button>
          <button
            type="button"
            class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
            onClick={props.onLoad}
          >
            Load Metadata
          </button>
          <Show when={props.results?.data?.length}>
            <button
              type="button"
              class="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-blue-300"
              disabled={props.saving}
              onClick={props.onSave}
            >
              {props.saving ? "Saving..." : "Save Metadata"}
            </button>
          </Show>
        </div>
      </div>

      <div class="mt-4 overflow-x-auto">
        <table class="min-w-full table-fixed divide-y divide-gray-200 text-sm">
          <thead class="bg-gray-50">
            <tr>
              <For each={tableColumns()}>
                {(column) => (
                  <th scope="col" class="px-4 py-2 text-left font-semibold text-gray-700">
                    {formatColumnName(column)}
                  </th>
                )}
              </For>
            </tr>
          </thead>
          <tbody class="divide-y divide-gray-100 bg-white">
            <Show
              when={!props.loading && props.results?.data?.length}
              fallback={
                <For each={placeholderRows}>
                  {() => (
                    <tr class="animate-pulse">
                      <For each={tableColumns()}>
                        {() => (
                          <td class="px-4 py-3">
                            <div class="max-w-xs overflow-x-auto whitespace-nowrap">
                              <SkeletonCell />
                            </div>
                          </td>
                        )}
                      </For>
                    </tr>
                  )}
                </For>
              }
            >
              <For each={props.results?.data || []}>
                {(row) => (
                  <tr class="hover:bg-gray-50">
                    <For each={tableColumns()}>
                      {(column) => {
                        const displayValue = stringifyValue(row[column]);
                        return (
                          <td class="px-4 py-2 align-top">
                            <div
                              class="max-w-xs overflow-x-auto whitespace-nowrap text-gray-800"
                              title={displayValue}
                            >
                              {displayValue}
                            </div>
                          </td>
                        );
                      }}
                    </For>
                  </tr>
                )}
              </For>
            </Show>
          </tbody>
        </table>
      </div>
    </section>
  );
}

