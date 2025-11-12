import { For, Show } from "solid-js";

interface SimulationStatus {
  simulation_id: string;
  status: string;
  progress: number;
  current_step: string;
  message: string;
  logs: string[];
  error?: string;
}

interface SimulationStatusDisplayProps {
  simulationId: string;
  status: SimulationStatus | null;
}

const SIMULATION_STEPS = [
  "Initializing",
  "Generating antenna configuration",
  "Computing max baseline",
  "Creating sky model",
  "Running interferometric simulation",
  "Processing results",
  "Saving output",
];

export function SimulationStatusDisplay(props: SimulationStatusDisplayProps) {
  return (
    <section class="bg-white rounded-lg shadow-md p-6 space-y-4">
      <div class="flex items-center justify-between">
        <h3 class="text-lg font-semibold text-gray-900">Simulation Status</h3>
        <Show when={props.status}>
          <span class={`px-3 py-1 rounded-full text-xs font-medium ${
            props.status?.status === 'completed' ? 'bg-green-100 text-green-800' :
            props.status?.status === 'failed' ? 'bg-red-100 text-red-800' :
            props.status?.status === 'running' ? 'bg-blue-100 text-blue-800' :
            'bg-gray-100 text-gray-800'
          }`}>
            {props.status?.status.toUpperCase() || 'CONNECTING'}
          </span>
        </Show>
      </div>
      
      <div class="text-sm text-gray-600">
        <p>ID: <code class="bg-gray-100 px-2 py-1 rounded text-xs">{props.simulationId}</code></p>
        <Show when={props.status}>
          <p class="mt-2">Current Step: <span class="font-medium">{props.status?.current_step || 'N/A'}</span></p>
          <p class="mt-1">Message: <span class="text-gray-800">{props.status?.message || 'N/A'}</span></p>
        </Show>
      </div>
      
      {/* Progress Bar */}
      <Show when={props.status}>
        <div class="space-y-2">
          <div class="flex items-center justify-between text-sm">
            <span class="text-gray-700">Progress</span>
            <span class="font-medium text-gray-900">{Math.round(props.status?.progress || 0)}%</span>
          </div>
          <div class="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
            <div 
              class="h-full bg-blue-600 transition-all duration-300 ease-out"
              style={`width: ${props.status?.progress || 0}%`}
            />
          </div>
        </div>
      </Show>
      
      {/* Simulation Steps */}
      <Show when={props.status}>
        <div class="space-y-2">
          <p class="text-sm font-medium text-gray-700">Simulation Steps:</p>
          <div class="space-y-1">
            <For each={SIMULATION_STEPS}>
              {(step, index) => {
                const currentStep = props.status?.current_step || '';
                const stepProgress = (index() + 1) / SIMULATION_STEPS.length * 100;
                const isActive = currentStep.toLowerCase().includes(step.toLowerCase()) || 
                               (props.status?.progress || 0) >= stepProgress;
                const isCompleted = (props.status?.progress || 0) > stepProgress;
                
                return (
                  <div class="flex items-center space-x-2 text-sm">
                    <div class={`w-4 h-4 rounded-full flex items-center justify-center ${
                      isCompleted ? 'bg-green-500' : isActive ? 'bg-blue-500' : 'bg-gray-300'
                    }`}>
                      {isCompleted && <span class="text-white text-xs">✓</span>}
                    </div>
                    <span class={isActive ? 'text-gray-900 font-medium' : 'text-gray-500'}>
                      {step}
                    </span>
                  </div>
                );
              }}
            </For>
          </div>
        </div>
        
        {/* Error Display */}
        <Show when={props.status?.error}>
          <div class="bg-red-50 border border-red-200 rounded-md p-3">
            <p class="text-sm font-medium text-red-800">Error:</p>
            <p class="text-sm text-red-700 mt-1">{props.status?.error}</p>
          </div>
        </Show>
        
        {/* Logs Window */}
        <div class="border border-gray-300 rounded-md bg-gray-900 text-gray-100 font-mono text-xs">
          <div class="bg-gray-800 px-3 py-2 flex items-center justify-between rounded-t-md">
            <span class="text-gray-300 font-medium">Backend Logs</span>
            <span class="text-gray-500 text-xs">{props.status?.logs?.length || 0} entries</span>
          </div>
          <div 
            class="p-3 max-h-64 overflow-y-auto space-y-1"
            ref={(el) => {
              // Auto-scroll to bottom when logs update
              if (el && props.status?.logs) {
                setTimeout(() => {
                  el.scrollTop = el.scrollHeight;
                }, 0);
              }
            }}
          >
            <Show 
              when={props.status?.logs && props.status!.logs.length > 0}
              fallback={<p class="text-gray-500">No logs yet...</p>}
            >
              <For each={props.status?.logs}>
                {(log) => (
                  <div class="text-gray-300 whitespace-pre-wrap break-words">
                    {log}
                  </div>
                )}
              </For>
            </Show>
          </div>
        </div>
      </Show>
      
      {/* Connecting message */}
      <Show when={!props.status}>
        <div class="text-center py-4 text-gray-500">
          <p>Connecting to simulation status...</p>
        </div>
      </Show>
    </section>
  );
}

