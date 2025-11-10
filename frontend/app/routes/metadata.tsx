import {
  For,
  Show,
  createMemo,
  createResource,
  createSignal,
  onMount,
  type JSX,
} from "solid-js";
import { metadataApi, type MetadataQuery, type MetadataResponse } from "../lib/api/metadata";

const bandOptions = [3, 4, 5, 6, 7, 8, 9, 10];
const placeholderRows = Array.from({ length: 5 }, (_, index) => index);
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

const RESULTS_CACHE_KEY = "almasim:metadata-results";
const DEFAULT_METADATA_PATH = "almasim/metadata";

type FileSystemFileHandle = {
  name?: string;
  createWritable: () => Promise<{ write: (data: Blob) => Promise<void>; close: () => Promise<void> }>;
};

type SaveFilePickerOptions = {
  suggestedName?: string;
  types?: { description?: string; accept: Record<string, string[]> }[];
};

const numberFromForm = (value: FormDataEntryValue | null) => {
  if (value === null || value === undefined || value === "") return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const rangeFromForm = (
  form: FormData,
  minKey: string,
  maxKey: string
): [number, number] | undefined => {
  const min = numberFromForm(form.get(minKey));
  const max = numberFromForm(form.get(maxKey));
  if (min === undefined || max === undefined) return undefined;
  return [min, max];
};

const FullScreenLoader = () => (
  <div class="fixed inset-0 z-40 flex flex-col items-center justify-center bg-white/90 text-gray-700">
    <div class="h-12 w-12 animate-spin rounded-full border-4 border-blue-200 border-t-blue-600" />
    <p class="mt-4 text-base font-medium">Contacting ALMA TAP...</p>
  </div>
);

const SkeletonCell = () => (
  <div class="h-4 w-full animate-pulse rounded bg-gray-200/80" aria-hidden="true" />
);

const supportsFilePicker = () =>
  typeof window !== "undefined" && typeof (window as Window & { showSaveFilePicker?: () => unknown }).showSaveFilePicker === "function";

const formatDateStamp = () => {
  const date = new Date();
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}-${String(
    date.getHours()
  ).padStart(2, "0")}${String(date.getMinutes()).padStart(2, "0")}`;
};

const getCachedResults = (): MetadataResponse | null => {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(RESULTS_CACHE_KEY);
    return raw ? (JSON.parse(raw) as MetadataResponse) : null;
  } catch {
    return null;
  }
};

const persistResults = (data: MetadataResponse) => {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(RESULTS_CACHE_KEY, JSON.stringify(data));
  } catch {
    /* ignore storage write failures */
  }
};

const downloadMetadata = (data: MetadataResponse, path: string) => {
  if (typeof window === "undefined") return;
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const safePath = path.replace(/^[.@/\\]+/, "");
  const sanitized = safePath.length > 0 ? safePath.replace(/\s+/g, "-").replace(/[^\w./-]/g, "_") : "";
  const segments = sanitized.split("/").filter(Boolean);
  let fileName = segments.length ? segments.join("_") : `metadata-results-${formatDateStamp()}`;
  if (!fileName.endsWith(".json")) fileName += ".json";
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(link.href);
};

const normalizeBackendPath = (input: string) => {
  const cleaned = input.replace(/^[./\\]+/, "").trim() || DEFAULT_METADATA_PATH;
  if (cleaned.endsWith(".json")) return cleaned;
  return `${cleaned.replace(/\/$/, "")}/metadata-results.json`;
};

export default function MetadataPage() {
  const [scienceTypes] = createResource(metadataApi.getScienceTypes);
  const [results, setResults] = createSignal<MetadataResponse | null>(null);
  const [loading, setLoading] = createSignal(false);
  const [saving, setSaving] = createSignal(false);
  const [error, setError] = createSignal<string>("");
  const [loadModalOpen, setLoadModalOpen] = createSignal(false);
  const [saveModalOpen, setSaveModalOpen] = createSignal(false);
  const [statusMessage, setStatusMessage] = createSignal("");

  const runQuery = async (query: MetadataQuery) => {
    setLoading(true);
    setError("");
    try {
      const data = await metadataApi.query(query);
      setResults(data);
      persistResults(data);
      setStatusMessage(`Fetched ${data.count} rows from ALMA TAP`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to fetch metadata");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit: JSX.EventHandlerUnion<HTMLFormElement, SubmitEvent> = async (event) => {
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

    const fovRange = rangeFromForm(formData, "fov_min", "fov_max");
    if (fovRange) query.fov_range = fovRange;

    const timeRange = rangeFromForm(formData, "time_min", "time_max");
    if (timeRange) query.time_resolution_range = timeRange;

    const freqRange = rangeFromForm(formData, "freq_min", "freq_max");
    if (freqRange) query.frequency_range = freqRange;

    await runQuery(query);
  };

  const parseCsv = (input: string): Array<Record<string, string>> => {
    const rows: string[][] = [];
    let current = "";
    let inQuotes = false;
    const currentRow: string[] = [];

    const pushValue = () => {
      currentRow.push(current);
      current = "";
    };

    const pushRow = () => {
      pushValue();
      rows.push([...currentRow]);
      currentRow.length = 0;
    };

    for (let i = 0; i < input.length; i++) {
      const char = input[i];
      if (char === '"') {
        if (inQuotes && input[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === "," && !inQuotes) {
        pushValue();
      } else if ((char === "\n" || char === "\r") && !inQuotes) {
        if (char === "\r" && input[i + 1] === "\n") i++;
        pushRow();
      } else {
        current += char;
      }
    }

    if (current.length > 0 || currentRow.length > 0) {
      pushRow();
    }

    if (!rows.length) return [];
    const [header, ...dataRows] = rows;
    return dataRows
      .filter((cells) => cells.some((cell) => cell.trim().length))
      .map((cells) =>
        header.reduce<Record<string, string>>((acc, column, index) => {
          acc[column] = cells[index] ?? "";
          return acc;
        }, {})
      );
  };

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

  const parseMetadataFile = async (file: File): Promise<MetadataResponse> => {
    const text = await file.text();
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (extension === "csv") {
      const data = parseCsv(text);
      return { data, count: data.length };
    }

    try {
      const parsed = JSON.parse(text) as Partial<MetadataResponse>;
      if (!parsed || !Array.isArray(parsed.data)) {
        throw new Error();
      }
      return {
        data: parsed.data,
        count:
          typeof parsed.count === "number"
            ? parsed.count
            : Array.isArray(parsed.data)
            ? parsed.data.length
            : 0,
      };
    } catch {
      throw new Error("Invalid metadata file. Provide JSON with a `data` array or a CSV file.");
    }
  };

  const [localSaveFileName, setLocalSaveFileName] = createSignal("");
  let loadFormRef: HTMLFormElement | undefined;
  let saveFormRef: HTMLFormElement | undefined;
  let localSaveHandle: FileSystemFileHandle | null = null;

  const handleLoadFile: JSX.EventHandlerUnion<HTMLFormElement, SubmitEvent> = async (event) => {
    event.preventDefault();
    const formElement = loadFormRef ?? (event.currentTarget as HTMLFormElement | undefined);
    if (!formElement) return;
    const formData = new FormData(formElement);

    const file = formData.get("file_upload");
    const filePath = (formData.get("file_path") as string)?.trim();

    if (!(file instanceof File) && !filePath) {
      setError("Select a metadata file or provide a backend-accessible path.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      let data: MetadataResponse;
      if (file instanceof File && file.size > 0) {
        data = await parseMetadataFile(file);
      } else if (filePath) {
        data = await metadataApi.load(filePath);
      } else {
        throw new Error("Unable to determine how to load the metadata.");
      }

      setResults(data);
      persistResults(data);
      setLoadModalOpen(false);
      formElement.reset();
      setStatusMessage(`Loaded ${data.count} rows from ${file instanceof File ? "local file" : filePath}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load metadata file");
    } finally {
      setLoading(false);
    }
  };

  const handleSaveMetadata: JSX.EventHandlerUnion<HTMLFormElement, SubmitEvent> = async (event) => {
    event.preventDefault();
    const current = results();
    if (!current || !current.data?.length) {
      setError("No metadata to save yet. Run a query or load data first.");
      return;
    }
    const formElement = saveFormRef ?? (event.currentTarget as HTMLFormElement | undefined);
    if (!formElement) return;
    const formData = new FormData(formElement);
    const backendPath = normalizeBackendPath((formData.get("save_path") as string) || DEFAULT_METADATA_PATH);

    setSaving(true);
    setError("");
    let backendSucceeded = false;
    try {
      await metadataApi.save({ path: backendPath, data: current.data });
      backendSucceeded = true;
      setStatusMessage(`Saved ${current.count ?? current.data.length} rows to backend path ${backendPath}`);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unable to save metadata via API. We'll keep working on a local copy.";
      setError(message);
    } finally {
      setSaving(false);
    }

    const completeCleanup = () => {
      formElement.reset();
      localSaveHandle = null;
      setLocalSaveFileName("");
      setSaveModalOpen(false);
    };

    const needsLocalFallback = !backendSucceeded;
    try {
      if (localSaveHandle) {
        const writable = await localSaveHandle.createWritable();
        await writable.write(new Blob([JSON.stringify(current, null, 2)], { type: "application/json" }));
        await writable.close();
        setStatusMessage(
          `Saved locally to ${localSaveFileName() || localSaveHandle.name || "metadata.json"}${
            backendSucceeded ? " (backend copy updated too)" : ""
          }`,
        );
      } else if (needsLocalFallback) {
        downloadMetadata(current, backendPath);
        setStatusMessage(`Downloaded metadata snapshot (${current.count ?? current.data.length} rows)`);
      }
    } catch (localError) {
      console.error(localError);
      setError("Unable to write metadata locally.");
    } finally {
      completeCleanup();
    }
  };

  const chooseLocalSavePath = async () => {
    if (!supportsFilePicker()) {
      setError("Your browser does not support choosing a local save location. The file will download instead.");
      return;
    }
    try {
      const picker = (window as Window & {
        showSaveFilePicker?: (options: SaveFilePickerOptions) => Promise<FileSystemFileHandle>;
      }).showSaveFilePicker;
      if (!picker) {
        setError("File picker API is unavailable in this environment.");
        return;
      }
      const handle = await picker({
        suggestedName: `almasim-metadata-${formatDateStamp()}.json`,
        types: [
          {
            description: "JSON file",
            accept: { "application/json": [".json"] },
          },
        ],
      });
      localSaveHandle = handle;
      setLocalSaveFileName(handle.name || "");
      setStatusMessage(`Local save target selected: ${handle.name || "metadata.json"}`);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        return;
      }
      setError(err instanceof Error ? err.message : "Unable to open save dialog.");
    }
  };

  const clearMetadata = () => {
    setResults(null);
    setStatusMessage("Metadata cleared.");
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(RESULTS_CACHE_KEY);
    }
  };

  const resultColumns = () => {
    const data = results()?.data;
    if (!data || data.length === 0) return [] as string[];
    return Object.keys(data[0]);
  };

  const tableColumns = createMemo(() => {
    const dynamic = resultColumns();
    return dynamic.length ? dynamic : defaultColumns;
  });

  const initialLoading = () => scienceTypes.state === "pending" && !scienceTypes();

  onMount(() => {
    const cached = getCachedResults();
    if (cached) {
      setResults(cached);
      setStatusMessage(`Loaded cached metadata (${cached.count ?? cached.data.length} rows)`);
    }
  });

  return (
    <div class="container mx-auto px-4 py-8">
      <Show when={initialLoading()}>
        <FullScreenLoader />
      </Show>
      <div class="max-w-6xl mx-auto space-y-8">
        <header>
          <h1 class="text-3xl font-bold text-gray-900">Metadata Explorer</h1>
          <p class="text-gray-600 mt-2">
            Query ALMA observation metadata or load a precomputed dataset.
          </p>
        </header>

        <section class="grid grid-cols-1 gap-6">
          <form
            onSubmit={handleSubmit}
            class="bg-white rounded-lg shadow p-6 space-y-4"
          >
            <div class="flex items-center justify-between">
              <h2 class="text-xl font-semibold text-gray-900">Query Builder</h2>
              <button
                type="submit"
                disabled={loading()}
                class="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-400"
              >
                {loading() ? "Running..." : "Run Query"}
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
                    when={scienceTypes()}
                    fallback={<option>Loading keywords...</option>}
                  >
                    <For each={scienceTypes()?.keywords || []}>
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
                    when={scienceTypes()}
                    fallback={<option>Loading categories...</option>}
                  >
                    <For each={scienceTypes()?.categories || []}>
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

        </section>

        <Show when={error()}>
          <div class="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800">
            {error()}
          </div>
        </Show>

        <Show when={statusMessage()}>
          <div class="rounded-md border border-blue-200 bg-blue-50 p-4 text-sm text-blue-800">
            {statusMessage()}
          </div>
        </Show>

        <section class="bg-white rounded-lg shadow p-6">
          <div class="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 class="text-xl font-semibold text-gray-900">Results</h2>
              <span class="text-sm text-gray-600">
                {results()?.count ? `${results()?.count} matches` : "No data yet"}
              </span>
            </div>
            <div class="flex flex-wrap gap-2">
              <button
                type="button"
                class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed"
                onClick={clearMetadata}
                disabled={loading() || !results()}
              >
                Clear Metadata
              </button>
              <button
                type="button"
                class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
                onClick={() => setLoadModalOpen(true)}
              >
                Load Metadata
              </button>
              <Show when={results()?.data?.length}>
                <button
                  type="button"
                  class="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-blue-300"
                  disabled={saving()}
                  onClick={() => setSaveModalOpen(true)}
                >
                  {saving() ? "Saving..." : "Save Metadata"}
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
                  when={!loading() && results()?.data?.length}
                  fallback={
                    <For each={placeholderRows}>
                      {(rowIndex) => (
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
                  <For each={results()?.data || []}>
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
        <Show when={loadModalOpen()}>
          <div
            class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4"
            role="dialog"
            aria-modal="true"
          >
            <div class="w-full max-w-lg rounded-lg bg-white shadow-xl">
              <header class="flex items-center justify-between border-b px-6 py-4">
                <h2 class="text-lg font-semibold text-gray-900">Load Metadata File</h2>
                <button
                  type="button"
                  class="text-gray-500 hover:text-gray-800"
                  aria-label="Close"
                  onClick={() => setLoadModalOpen(false)}
                >
                  ✕
                </button>
              </header>
              <form ref={(el) => (loadFormRef = el)} onSubmit={handleLoadFile} class="space-y-4 px-6 py-6">
                <div class="space-y-1">
                  <label class="block text-sm font-medium text-gray-700" for="file_upload">
                    Select local file (JSON or CSV)
                  </label>
                  <input
                    id="file_upload"
                    name="file_upload"
                    type="file"
                    accept=".json,.csv,application/json,text/csv"
                    class="w-full rounded-md border border-gray-300 p-2 text-sm text-gray-700 file:mr-4 file:rounded file:border-0 file:bg-gray-100 file:px-4 file:py-2 file:text-sm file:font-medium file:text-gray-700 hover:file:bg-gray-200"
                    autofocus
                  />
                  <p class="text-xs text-gray-500">
                    The file contents stay on your machine; we parse supported formats directly in the browser.
                  </p>
                </div>

                <div class="space-y-1">
                  <label class="block text-sm font-medium text-gray-700" for="file_path">
                    Or load from server path
                  </label>
                  <input
                    type="text"
                    id="file_path"
                    name="file_path"
                    placeholder="data/metadata/sample.json"
                    defaultValue={DEFAULT_METADATA_PATH}
                    class="w-full rounded-md border border-gray-300 p-2"
                  />
                  <p class="text-xs text-gray-500">
                    Use when the backend already has access to the metadata file.
                  </p>
                </div>

                <div class="flex items-center justify-end gap-2">
                  <button
                    type="button"
                    class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
                    onClick={() => setLoadModalOpen(false)}
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={loading()}
                    class="rounded-md bg-gray-900 px-4 py-2 text-sm font-medium text-white hover:bg-gray-800 disabled:bg-gray-600"
                  >
                    {loading() ? "Loading..." : "Load"}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </Show>

        <Show when={saveModalOpen()}>
          <div
            class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4"
            role="dialog"
            aria-modal="true"
          >
            <div class="w-full max-w-lg rounded-lg bg-white shadow-xl">
              <header class="flex items-center justify-between border-b px-6 py-4">
                <h2 class="text-lg font-semibold text-gray-900">Save Metadata</h2>
                <button
                  type="button"
                  class="text-gray-500 hover:text-gray-800"
                  aria-label="Close"
                  onClick={() => setSaveModalOpen(false)}
                >
                  ✕
                </button>
              </header>
              <form ref={(el) => (saveFormRef = el)} onSubmit={handleSaveMetadata} class="space-y-4 px-6 py-6">
                <p class="text-sm text-gray-600">
                  Save the current metadata result set to a backend-visible path (default:{" "}
                  <code class="rounded bg-gray-100 px-1 py-0.5 text-xs text-gray-800">{DEFAULT_METADATA_PATH}</code>) or pick a local
                  destination on your machine.
                </p>
                <label class="block text-sm font-medium text-gray-700" for="save_path">
                  Destination path
                  <input
                    type="text"
                    id="save_path"
                    name="save_path"
                    defaultValue={DEFAULT_METADATA_PATH}
                    placeholder={DEFAULT_METADATA_PATH}
                    class="mt-1 w-full rounded-md border border-gray-300 p-2"
                  />
                </label>
                <div class="text-xs text-gray-500">
                  We attempt to persist the file server-side; if that fails, a download starts in your browser.
                </div>

                <div class="rounded-md border border-dashed border-gray-300 p-3">
                  <div class="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p class="text-sm font-medium text-gray-700">Local save location (optional)</p>
                      <p class="text-xs text-gray-500">
                        {supportsFilePicker()
                          ? localSaveFileName()
                            ? `Selected: ${localSaveFileName()}`
                            : "No local file chosen yet."
                          : "Your browser will prompt for a download instead."}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={chooseLocalSavePath}
                      disabled={!supportsFilePicker()}
                      class="rounded-md border border-gray-300 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed disabled:text-gray-400"
                    >
                      Choose location
                    </button>
                  </div>
                </div>

                <div class="flex items-center justify-end gap-2">
                  <button
                    type="button"
                    class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
                    onClick={() => setSaveModalOpen(false)}
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={saving()}
                    class="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-blue-300"
                  >
                    {saving() ? "Saving..." : "Save"}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </Show>
      </div>
    </div>
  );
}
