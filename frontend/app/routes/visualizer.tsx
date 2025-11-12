import { createSignal, Show, createResource } from "solid-js";
import { DatacubeFileList } from "../components/visualizer/DatacubeFileList";
import { DatacubeUpload } from "../components/visualizer/DatacubeUpload";
import { ImageCanvas } from "../components/visualizer/ImageCanvas";
import { ImageStatistics } from "../components/visualizer/ImageStatistics";

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface ImageData {
  image: number[][];
  stats: {
    shape: number[];
    integrated_shape: number[];
    min: number;
    max: number;
    mean: number;
    std: number;
    cube_name: string;
  };
  method: string;
}

interface DatacubeFile {
  name: string;
  path: string;
  size: number;
  modified: number;
}

interface FileListResponse {
  files: DatacubeFile[];
  output_dir: string;
}

export default function Visualizer() {
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [imageData, setImageData] = createSignal<ImageData | null>(null);
  const [integrationMethod, setIntegrationMethod] = createSignal<"sum" | "mean">("sum");
  const [outputDir, setOutputDir] = createSignal<string>("");
  
  // Canvas and zoom/pan state
  const [scale, setScale] = createSignal(1.0);
  const [panX, setPanX] = createSignal(0);
  const [panY, setPanY] = createSignal(0);

  // Load file list
  const [fileList, { refetch: refetchFiles }] = createResource(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/visualizer/files`);
      if (!response.ok) throw new Error("Failed to load file list");
      const data: FileListResponse = await response.json();
      setOutputDir(data.output_dir);
      return data;
    } catch (err) {
      console.error("Failed to load file list:", err);
      return { files: [], output_dir: "" };
    }
  });

  // Process a file (either uploaded or from server)
  const processFile = async (file: File | string) => {
    setLoading(true);
    setError(null);

    try {
      let formData: FormData;
      
      if (typeof file === "string") {
        // Load file from server
        const fileResponse = await fetch(`${API_BASE_URL}/api/v1/visualizer/files/${encodeURIComponent(file)}`);
        if (!fileResponse.ok) {
          throw new Error(`Failed to load file: ${fileResponse.statusText}`);
        }
        const blob = await fileResponse.blob();
        const serverFile = new File([blob], file.split("/").pop() || "file.npz", { type: "application/octet-stream" });
        formData = new FormData();
        formData.append("file", serverFile);
      } else {
        // Use uploaded file
        formData = new FormData();
        formData.append("file", file);
      }
      
      formData.append("method", integrationMethod());

      const response = await fetch(`${API_BASE_URL}/api/v1/visualizer/integrate`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data: ImageData = await response.json();
      setImageData(data);
      setScale(1.0);
      setPanX(0);
      setPanY(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to process datacube");
    } finally {
      setLoading(false);
    }
  };

  // Handle file upload
  const handleFileUpload = async (event: Event) => {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    if (!file.name.endsWith(".npz")) {
      setError("Please select a .npz file");
      return;
    }

    await processFile(file);
  };

  // Handle file selection from list
  const handleFileSelect = async (filePath: string) => {
    await processFile(filePath);
  };

  const handleReset = () => {
    setScale(1.0);
    setPanX(0);
    setPanY(0);
  };

  return (
    <div class="container mx-auto px-4 py-8">
      <div class="max-w-6xl mx-auto space-y-6">
        <div class="bg-white rounded-lg shadow-md p-6">
          <h1 class="text-2xl font-bold text-gray-900 mb-4">Datacube Visualizer</h1>
          
          <div class="space-y-4">
            <DatacubeFileList
              files={fileList()?.files || []}
              loading={fileList.loading}
              outputDir={outputDir()}
              onFileSelect={handleFileSelect}
              onRefresh={refetchFiles}
              disabled={loading()}
            />

            <DatacubeUpload
              onFileUpload={handleFileUpload}
              integrationMethod={integrationMethod()}
              onMethodChange={setIntegrationMethod}
              loading={loading()}
            />

            <Show when={error()}>
              <div class="bg-red-50 border border-red-200 rounded-md p-3">
                <p class="text-sm text-red-800">{error()}</p>
              </div>
            </Show>

            <Show when={loading()}>
              <div class="text-center py-4">
                <p class="text-gray-600">Processing datacube...</p>
              </div>
            </Show>

            <Show when={imageData()}>
              <ImageStatistics
                stats={imageData()!.stats}
                method={imageData()!.method}
              />
            </Show>
          </div>
        </div>

        <Show when={imageData()}>
          <div class="bg-white rounded-lg shadow-md p-6">
            <ImageCanvas
              imageData={imageData()}
              scale={scale()}
              panX={panX()}
              panY={panY()}
              onScaleChange={setScale}
              onPanChange={(x, y) => {
                setPanX(x);
                setPanY(y);
              }}
              onReset={handleReset}
            />
          </div>
        </Show>
      </div>
    </div>
  );
}
