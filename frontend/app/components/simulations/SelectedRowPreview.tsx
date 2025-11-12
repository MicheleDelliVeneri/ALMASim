import { Show } from "solid-js";

interface SelectedRowPreviewProps {
  row: Record<string, unknown> | null;
  getRowValue: (row: Record<string, unknown>, key: string) => string;
  getRowNumber: (row: Record<string, unknown>, key: string) => number | null;
  sourceType: string;
  nPix: number;
  nChannels: number;
}

export function SelectedRowPreview(props: SelectedRowPreviewProps) {
  if (!props.row) return null;

  return (
    <section class="bg-blue-50 rounded-lg shadow-md p-6 border border-blue-200">
      <h3 class="text-lg font-semibold text-gray-900 mb-4">Selected Row Preview</h3>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div>
          <span class="font-medium text-gray-700">Source:</span>
          <p class="text-gray-900 mt-1">
            {props.getRowValue(props.row, "ALMA_source_name") || props.getRowValue(props.row, "source_name")}
          </p>
        </div>
        <div>
          <span class="font-medium text-gray-700">RA:</span>
          <p class="text-gray-900 mt-1">
            {(() => {
              const val = props.getRowNumber(props.row, "RA") ?? props.getRowNumber(props.row, "ra");
              return val !== null ? val.toFixed(6) : "N/A";
            })()}
          </p>
        </div>
        <div>
          <span class="font-medium text-gray-700">Dec:</span>
          <p class="text-gray-900 mt-1">
            {(() => {
              const val = props.getRowNumber(props.row, "Dec") ?? props.getRowNumber(props.row, "dec");
              return val !== null ? val.toFixed(6) : "N/A";
            })()}
          </p>
        </div>
        <div>
          <span class="font-medium text-gray-700">Band:</span>
          <p class="text-gray-900 mt-1">
            {props.getRowValue(props.row, "Band") || props.getRowValue(props.row, "band")}
          </p>
        </div>
        <div>
          <span class="font-medium text-gray-700">FOV:</span>
          <p class="text-gray-900 mt-1">
            {(() => {
              const val = props.getRowNumber(props.row, "FOV") ?? props.getRowNumber(props.row, "fov");
              return val !== null ? `${val.toFixed(2)} arcsec` : "N/A";
            })()}
          </p>
        </div>
        <div>
          <span class="font-medium text-gray-700">Frequency:</span>
          <p class="text-gray-900 mt-1">
            {(() => {
              const val = props.getRowNumber(props.row, "Freq") ?? props.getRowNumber(props.row, "freq");
              return val !== null ? `${val.toFixed(2)} GHz` : "N/A";
            })()}
          </p>
        </div>
        <div>
          <span class="font-medium text-gray-700">Source Type:</span>
          <p class="text-gray-900 mt-1">{props.sourceType}</p>
        </div>
        <div>
          <span class="font-medium text-gray-700">Cube Size:</span>
          <p class="text-gray-900 mt-1">
            {props.nPix} × {props.nPix} × {props.nChannels}
          </p>
        </div>
      </div>
    </section>
  );
}

