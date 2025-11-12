import { For, Show } from "solid-js";
import type { MetadataResponse } from "../../lib/api/metadata";

interface MetadataSelectorProps {
  metadata: MetadataResponse | null;
  selectedIndex: number | null;
  onSelect: (index: number) => void;
  getRowValue: (row: Record<string, unknown>, key: string) => string;
  getRowNumber: (row: Record<string, unknown>, key: string) => number | null;
}

export function MetadataSelector(props: MetadataSelectorProps) {
  return (
    <section class="bg-white rounded-lg shadow-md p-6">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h2 class="text-xl font-semibold text-gray-900">Select Metadata Row</h2>
          <p class="text-sm text-gray-600 mt-1">
            {props.metadata?.count 
              ? `${props.metadata?.count} rows available` 
              : "No metadata loaded. Go to Metadata page to query or load data."}
          </p>
        </div>
        <a
          href="/metadata"
          class="text-blue-600 hover:text-blue-800 text-sm font-medium"
        >
          Go to Metadata →
        </a>
      </div>

      <Show
        when={props.metadata?.data && props.metadata!.data.length > 0}
        fallback={
          <div class="text-center py-8 text-gray-500">
            <p>No metadata available. Please query or load metadata first.</p>
            <a href="/metadata" class="text-blue-600 hover:text-blue-800 mt-2 inline-block">
              Go to Metadata page
            </a>
          </div>
        }
      >
        <div class="overflow-x-auto">
          <table class="min-w-full divide-y divide-gray-200 text-sm">
            <thead class="bg-gray-50">
              <tr>
                <th class="px-4 py-2 text-left font-semibold text-gray-700 w-12">Select</th>
                <th class="px-4 py-2 text-left font-semibold text-gray-700">Source Name</th>
                <th class="px-4 py-2 text-left font-semibold text-gray-700">RA (deg)</th>
                <th class="px-4 py-2 text-left font-semibold text-gray-700">Dec (deg)</th>
                <th class="px-4 py-2 text-left font-semibold text-gray-700">Band</th>
                <th class="px-4 py-2 text-left font-semibold text-gray-700">FOV (arcsec)</th>
                <th class="px-4 py-2 text-left font-semibold text-gray-700">Freq (GHz)</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-100 bg-white">
              <For each={props.metadata?.data || []}>
                {(row, index) => {
                  const isSelected = () => props.selectedIndex === index();
                  return (
                    <tr
                      class={`cursor-pointer transition-colors ${
                        isSelected()
                          ? "bg-blue-50 hover:bg-blue-100"
                          : "hover:bg-gray-50"
                      }`}
                      onClick={() => props.onSelect(index())}
                    >
                      <td class="px-4 py-2">
                        <input
                          type="radio"
                          name="selected_row"
                          checked={isSelected()}
                          onChange={() => props.onSelect(index())}
                          class="text-blue-600"
                        />
                      </td>
                      <td class="px-4 py-2 text-gray-800">
                        {props.getRowValue(row, "ALMA_source_name") || props.getRowValue(row, "source_name")}
                      </td>
                      <td class="px-4 py-2 text-gray-800">
                        {(() => {
                          const val = props.getRowNumber(row, "RA") ?? props.getRowNumber(row, "ra");
                          return val !== null ? val.toFixed(4) : "N/A";
                        })()}
                      </td>
                      <td class="px-4 py-2 text-gray-800">
                        {(() => {
                          const val = props.getRowNumber(row, "Dec") ?? props.getRowNumber(row, "dec");
                          return val !== null ? val.toFixed(4) : "N/A";
                        })()}
                      </td>
                      <td class="px-4 py-2 text-gray-800">
                        {props.getRowValue(row, "Band") || props.getRowValue(row, "band")}
                      </td>
                      <td class="px-4 py-2 text-gray-800">
                        {(() => {
                          const val = props.getRowNumber(row, "FOV") ?? props.getRowNumber(row, "fov");
                          return val !== null ? val.toFixed(2) : "N/A";
                        })()}
                      </td>
                      <td class="px-4 py-2 text-gray-800">
                        {(() => {
                          const val = props.getRowNumber(row, "Freq") ?? props.getRowNumber(row, "freq");
                          return val !== null ? val.toFixed(2) : "N/A";
                        })()}
                      </td>
                    </tr>
                  );
                }}
              </For>
            </tbody>
          </table>
        </div>
      </Show>
    </section>
  );
}

