import { createSignal } from "solid-js";
import { simulationApi, type SimulationParamsCreate } from "../lib/api/simulation";

export default function Simulations() {
  const [loading, setLoading] = createSignal(false);
  const [message, setMessage] = createSignal("");
  const [simulationId, setSimulationId] = createSignal("");

  const handleSubmit = async (event: Event) => {
    event.preventDefault();
    setLoading(true);
    setMessage("");

    const formData = new FormData(event.target as HTMLFormElement);
    const params: SimulationParamsCreate = {
      idx: 0,
      source_name: formData.get("source_name") as string,
      member_ouid: formData.get("member_ouid") as string,
      project_name: formData.get("project_name") as string,
      ra: parseFloat(formData.get("ra") as string),
      dec: parseFloat(formData.get("dec") as string),
      band: parseFloat(formData.get("band") as string),
      ang_res: parseFloat(formData.get("ang_res") as string),
      vel_res: parseFloat(formData.get("vel_res") as string),
      fov: parseFloat(formData.get("fov") as string),
      obs_date: formData.get("obs_date") as string,
      pwv: parseFloat(formData.get("pwv") as string),
      int_time: parseFloat(formData.get("int_time") as string),
      bandwidth: parseFloat(formData.get("bandwidth") as string),
      freq: parseFloat(formData.get("freq") as string),
      freq_support: formData.get("freq_support") as string,
      cont_sens: parseFloat(formData.get("cont_sens") as string),
      antenna_array: formData.get("antenna_array") as string,
      source_type: (formData.get("source_type") as string) || "point",
      main_dir: (formData.get("main_dir") as string) || "./almasim",
      output_dir: (formData.get("output_dir") as string) || "./outputs",
      tng_dir: (formData.get("tng_dir") as string) || "./data/TNG100-1",
      galaxy_zoo_dir: (formData.get("galaxy_zoo_dir") as string) || "./data/galaxy_zoo",
      hubble_dir: (formData.get("hubble_dir") as string) || "./data/hubble",
    };

    try {
      const response = await simulationApi.create(params);
      setSimulationId(response.simulation_id);
      setMessage(`Simulation created successfully! ID: ${response.simulation_id}`);
    } catch (error) {
      setMessage(`Error: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div class="container mx-auto px-4 py-8">
      <div class="max-w-4xl mx-auto">
        <h1 class="text-3xl font-bold text-gray-900 mb-6">Create Simulation</h1>

        {message() && (
          <div
            class={`mb-4 p-4 rounded-lg ${
              message().includes("Error")
                ? "bg-red-100 text-red-800"
                : "bg-green-100 text-green-800"
            }`}
          >
            {message()}
          </div>
        )}

        <form onSubmit={handleSubmit} class="bg-white rounded-lg shadow-md p-6 space-y-6">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label for="source_name" class="block text-sm font-medium text-gray-700 mb-1">
                Source Name
              </label>
              <input
                type="text"
                id="source_name"
                name="source_name"
                required
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label for="member_ouid" class="block text-sm font-medium text-gray-700 mb-1">
                Member OUS UID
              </label>
              <input
                type="text"
                id="member_ouid"
                name="member_ouid"
                required
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label for="ra" class="block text-sm font-medium text-gray-700 mb-1">
                RA (deg)
              </label>
              <input
                type="number"
                id="ra"
                name="ra"
                step="0.0001"
                required
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label for="dec" class="block text-sm font-medium text-gray-700 mb-1">
                Dec (deg)
              </label>
              <input
                type="number"
                id="dec"
                name="dec"
                step="0.0001"
                required
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label for="source_type" class="block text-sm font-medium text-gray-700 mb-1">
                Source Type
              </label>
              <select
                id="source_type"
                name="source_type"
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="point">Point</option>
                <option value="gaussian">Gaussian</option>
                <option value="extended">Extended</option>
                <option value="diffuse">Diffuse</option>
                <option value="galaxy-zoo">Galaxy Zoo</option>
                <option value="molecular">Molecular</option>
                <option value="hubble-100">Hubble</option>
              </select>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading()}
            class="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {loading() ? "Creating..." : "Create Simulation"}
          </button>
        </form>
      </div>
    </div>
  );
}

