import { Show, createResource, createSignal, onMount, type JSX } from "solid-js";
import { metadataApi, type MetadataQuery, type MetadataResponse } from "../lib/api/metadata";
import { MetadataQueryForm } from "../components/metadata/MetadataQueryForm";
import { MetadataResultsTable } from "../components/metadata/MetadataResultsTable";
import { MetadataLoadModal } from "../components/metadata/MetadataLoadModal";
import { MetadataSaveModal } from "../components/metadata/MetadataSaveModal";
import { FullScreenLoader } from "../components/shared/Loader";
import { formatDateStamp, supportsFilePicker } from "../components/shared/utils";

const RESULTS_CACHE_KEY = "almasim:metadata-results";
const DEFAULT_METADATA_PATH = "data";

type FileSystemFileHandle = {
  name?: string;
  createWritable: () => Promise<{ write: (data: Blob) => Promise<void>; close: () => Promise<void> }>;
};

type SaveFilePickerOptions = {
  suggestedName?: string;
  types?: { description?: string; accept: Record<string, string[]> }[];
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

export default function MetadataPage() {
  const [scienceTypes] = createResource(metadataApi.getScienceTypes);
  const [results, setResults] = createSignal<MetadataResponse | null>(null);
  const [loading, setLoading] = createSignal(false);
  const [saving, setSaving] = createSignal(false);
  const [error, setError] = createSignal<string>("");
  const [loadModalOpen, setLoadModalOpen] = createSignal(false);
  const [saveModalOpen, setSaveModalOpen] = createSignal(false);
  const [statusMessage, setStatusMessage] = createSignal("");
  const [localSaveFileName, setLocalSaveFileName] = createSignal("");
  let loadFormRef: HTMLFormElement | undefined;
  let saveFormRef: HTMLFormElement | undefined;
  let localSaveHandle: FileSystemFileHandle | null = null;

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

        <MetadataQueryForm
          scienceTypes={scienceTypes() || null}
          loading={loading()}
          onSubmit={runQuery}
        />

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

        <MetadataResultsTable
          results={results()}
          loading={loading()}
          onClear={clearMetadata}
          onLoad={() => setLoadModalOpen(true)}
          onSave={() => setSaveModalOpen(true)}
          saving={saving()}
        />

        <MetadataLoadModal
          open={loadModalOpen()}
          loading={loading()}
          defaultPath={DEFAULT_METADATA_PATH}
          onClose={() => setLoadModalOpen(false)}
          onSubmit={handleLoadFile}
          formRef={(el) => (loadFormRef = el)}
        />

        <MetadataSaveModal
          open={saveModalOpen()}
          saving={saving()}
          defaultPath={DEFAULT_METADATA_PATH}
          localFileName={localSaveFileName()}
          onClose={() => setSaveModalOpen(false)}
          onSubmit={handleSaveMetadata}
          onChooseLocalPath={chooseLocalSavePath}
          formRef={(el) => (saveFormRef = el)}
        />
      </div>
    </div>
  );
}
