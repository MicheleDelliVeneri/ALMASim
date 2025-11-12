import { Show, createSignal, onMount, onCleanup } from "solid-js";
import { simulationApi, type SimulationParamsCreate } from "../lib/api/simulation";
import type { MetadataResponse } from "../lib/api/metadata";
import { ComputeBackendConfig } from "../components/simulations/ComputeBackendConfig";
import { SimulationForm } from "../components/simulations/SimulationForm";
import { MetadataSelector } from "../components/simulations/MetadataSelector";
import { SelectedRowPreview } from "../components/simulations/SelectedRowPreview";
import { SimulationStatusDisplay } from "../components/simulations/SimulationStatusDisplay";

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface SimulationStatus {
  simulation_id: string;
  status: string;
  progress: number;
  current_step: string;
  message: string;
  logs: string[];
  error?: string;
}

const RESULTS_CACHE_KEY = "almasim:metadata-results";

const getCachedResults = (): MetadataResponse | null => {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(RESULTS_CACHE_KEY);
    return raw ? (JSON.parse(raw) as MetadataResponse) : null;
  } catch {
    return null;
  }
};

export default function Simulations() {
  const [loading, setLoading] = createSignal(false);
  const [message, setMessage] = createSignal("");
  const [simulationId, setSimulationId] = createSignal("");
  const [metadata, setMetadata] = createSignal<MetadataResponse | null>(null);
  const [selectedRowIndex, setSelectedRowIndex] = createSignal<number | null>(null);
  
  // Simulation status tracking
  const [simulationStatus, setSimulationStatus] = createSignal<SimulationStatus | null>(null);
  const [ws, setWs] = createSignal<WebSocket | null>(null);
  
  // Cube dimensions
  const [nPix, setNPix] = createSignal<number>(256);
  const [nChannels, setNChannels] = createSignal<number>(128);
  
  // Source type
  const [sourceType, setSourceType] = createSignal<string>("point");
  
  // Compute backend configuration
  const [computeBackend, setComputeBackend] = createSignal<string>("local");
  const [backendConfig, setBackendConfig] = createSignal<Record<string, unknown>>({});

  // Load cached metadata on mount
  onMount(() => {
    const cached = getCachedResults();
    if (cached) {
      setMetadata(cached);
    }
  });
  
  // Cleanup WebSocket on unmount
  onCleanup(() => {
    const socket = ws();
    if (socket) {
      socket.close();
    }
  });
  
  // Connect to WebSocket when simulationId is set
  const connectWebSocket = (id: string) => {
    const wsUrl = API_BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://');
    const socket = new WebSocket(`${wsUrl}/api/v1/simulations/${id}/ws`);
    
    socket.onopen = () => {
      console.log('WebSocket connected');
    };
    
    socket.onmessage = (event) => {
      try {
        const status: SimulationStatus = JSON.parse(event.data);
        setSimulationStatus(status);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };
    
    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    socket.onclose = () => {
      console.log('WebSocket disconnected');
      setWs(null);
    };
    
    setWs(socket);
  };

  const selectedRow = () => {
    const idx = selectedRowIndex();
    const data = metadata()?.data;
    if (idx === null || !data || idx < 0 || idx >= data.length) return null;
    return data[idx];
  };

  // Helper to safely get and format values from metadata rows
  const getRowValue = (row: Record<string, unknown>, key: string): string => {
    const value = row[key];
    if (value === null || value === undefined || value === "") return "N/A";
    if (typeof value === "number") {
      if (isNaN(value)) return "N/A";
      return value.toString();
    }
    if (Array.isArray(value)) return value.join(", ");
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  };

  const getRowNumber = (row: Record<string, unknown>, key: string): number | null => {
    const value = row[key];
    if (value === null || value === undefined || value === "") return null;
    if (typeof value === "number") {
      if (isNaN(value)) return null;
      return value;
    }
    if (typeof value === "string") {
      const parsed = parseFloat(value);
      return isNaN(parsed) ? null : parsed;
    }
    return null;
  };

  const handleSubmit = async (event: Event) => {
    event.preventDefault();
    setLoading(true);
    setMessage("");

    const row = selectedRow();
    if (!row) {
      setMessage("Error: Please select a metadata row first.");
      setLoading(false);
      return;
    }

    // Extract values from metadata row with fallbacks
    // Try multiple key variations (PascalCase, camelCase, snake_case)
    const getValue = (keys: string[], fallback: any = null) => {
      for (const key of keys) {
        const value = row[key];
        if (value !== null && value !== undefined && value !== "") {
          return value;
        }
      }
      return fallback;
    };

    const getNumber = (keys: string[], fallback: number): number => {
      const value = getValue(keys, fallback);
      if (typeof value === "number") {
        if (isNaN(value)) return fallback;
        return value;
      }
      if (typeof value === "string") {
        const parsed = parseFloat(value);
        return isNaN(parsed) ? fallback : parsed;
      }
      return fallback;
    };

    const getString = (keys: string[], fallback: string): string => {
      const value = getValue(keys, fallback);
      return String(value ?? fallback);
    };

    const params: SimulationParamsCreate = {
      idx: 0,
      source_name: getString(["ALMA_source_name", "source_name"], "Unknown"),
      member_ouid: getString(["member_ous_uid", "member_ouid"], "unknown"),
      project_name: getString(["proposal_id", "project_name"], "ALMASim"),
      ra: getNumber(["RA", "ra"], 0.0),
      dec: getNumber(["Dec", "dec"], 0.0),
      band: getNumber(["Band", "band"], 3),
      ang_res: getNumber(["Ang.res.", "Ang.res", "ang_res"], 0.1),
      vel_res: getNumber(["Vel.res.", "Vel.res", "vel_res"], 1.0),
      fov: getNumber(["FOV", "fov"], 10.0),
      obs_date: getString(["Obs.date", "obs_date"], new Date().toISOString().split("T")[0]),
      pwv: getNumber(["PWV", "pwv"], 1.0),
      int_time: getNumber(["Int.Time", "int_time"], 3600.0),
      bandwidth: getNumber(["Bandwidth", "bandwidth"], 2.0),
      freq: getNumber(["Freq", "freq"], 100.0),
      freq_support: getString(["Freq.sup.", "Freq.sup", "freq_support"], "100.0-102.0"),
      cont_sens: getNumber(["Cont_sens_mJybeam", "Cont_sens", "cont_sens"], 0.1),
      antenna_array: getString(["antenna_arrays", "antenna_array"], "C43-1"),
      source_type: sourceType(),
      n_pix: nPix(),
      n_channels: nChannels(),
      main_dir: "./src/almasim",
      output_dir: "./outputs",
      tng_dir: "./data/TNG100-1",
      galaxy_zoo_dir: "./data/galaxy_zoo",
      hubble_dir: "./data/hubble",
      compute_backend: computeBackend(),
      compute_backend_config: backendConfig(),
    };

    try {
      const response = await simulationApi.create(params);
      setSimulationId(response.simulation_id);
      connectWebSocket(response.simulation_id);
      setMessage(`Simulation created successfully! ID: ${response.simulation_id}`);
    } catch (error) {
      setMessage(`Error: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div class="container mx-auto px-4 py-8">
      <div class="max-w-6xl mx-auto space-y-6">
        <header>
          <h1 class="text-3xl font-bold text-gray-900">Create Simulation</h1>
          <p class="text-gray-600 mt-2">
            Select a metadata row and configure simulation parameters.
          </p>
        </header>

        <Show when={message()}>
          <div
            class={`p-4 rounded-lg ${
              message().includes("Error")
                ? "bg-red-100 text-red-800 border border-red-200"
                : "bg-green-100 text-green-800 border border-green-200"
            }`}
          >
            {message()}
          </div>
        </Show>

        <ComputeBackendConfig
          backendType={computeBackend()}
          backendConfig={backendConfig()}
          onBackendTypeChange={setComputeBackend}
          onConfigChange={setBackendConfig}
        />

        <SimulationForm
          sourceType={sourceType()}
          nPix={nPix()}
          nChannels={nChannels()}
          onSourceTypeChange={setSourceType}
          onNPixChange={setNPix}
          onNChannelsChange={setNChannels}
        />

        <MetadataSelector
          metadata={metadata()}
          selectedIndex={selectedRowIndex()}
          onSelect={setSelectedRowIndex}
          getRowValue={getRowValue}
          getRowNumber={getRowNumber}
        />

        <SelectedRowPreview
          row={selectedRow()}
          getRowValue={getRowValue}
          getRowNumber={getRowNumber}
          sourceType={sourceType()}
          nPix={nPix()}
          nChannels={nChannels()}
        />

        <form onSubmit={handleSubmit}>
          <button
            type="submit"
            disabled={loading() || !selectedRow()}
            class="w-full bg-blue-600 text-white py-3 px-6 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-medium text-lg transition-colors"
          >
            {loading() ? "Creating Simulation..." : "Create Simulation"}
          </button>
          <p class="text-xs text-gray-500 mt-2 text-center">
            {!selectedRow() 
              ? "Please select a metadata row above" 
              : `Will create ${sourceType()} simulation with ${nPix()}×${nPix()}×${nChannels()} cube`}
          </p>
        </form>

        <Show when={simulationId()}>
          <SimulationStatusDisplay
            simulationId={simulationId()}
            status={simulationStatus()}
          />
        </Show>
      </div>
    </div>
  );
}
