/** Simulation API types and functions */
import { apiClient } from './client';

export interface ObservationConfigInput {
	name: string;
	array_type: '12m' | '7m' | 'TP';
	antenna_array: string;
	total_time_s: number;
	correlator?: string;
}

export interface SimulationParams {
	source_name: string;
	member_ouid: string;
	project_name: string;
	ra: number;
	dec: number;
	band: number;
	ang_res: number;
	vel_res: number;
	fov: number;
	obs_date: string;
	pwv: number;
	int_time: number;
	bandwidth: number;
	freq: number;
	freq_support: string;
	cont_sens: number;
	antenna_array: string;
	observation_configs?: ObservationConfigInput[];
	source_type?: string;
	n_pix?: number;
	n_channels?: number;
	tng_api_key?: string;
	ncpu?: number;
	rest_frequency?: unknown;
	redshift?: number;
	lum_infrared?: number;
	snr?: number;
	n_lines?: number;
	line_names?: unknown;
	save_mode?: string;
	inject_serendipitous?: boolean;
	robust?: number;
	compute_backend?: string;
	compute_backend_config?: Record<string, unknown>;
	ground_temperature_k?: number;
	correlator?: string;
	elevation_deg?: number;
}

export interface SimulationParamsCreate extends SimulationParams {
	idx: number;
	main_dir: string;
	output_dir: string;
	tng_dir: string;
	galaxy_zoo_dir: string;
	hubble_dir: string;
}

export interface SimulationResponse {
	simulation_id: string;
	status: string;
	output_path?: string;
	message?: string;
}

export interface SimulationStatus {
	simulation_id: string;
	status: string;
	progress?: number;
	message?: string;
}

export interface DaskTestResult {
	ok: boolean;
	scheduler: string;
	dashboard_port: number;
	workers: number;
	total_threads: number;
	total_memory_gb: number;
	error?: string | null;
}

export const simulationApi = {
	create: async (params: SimulationParamsCreate): Promise<SimulationResponse> => {
		return apiClient.post<SimulationResponse>('/api/v1/simulations/', params);
	},

	getStatus: async (simulationId: string): Promise<SimulationStatus> => {
		return apiClient.get<SimulationStatus>(`/api/v1/simulations/${simulationId}/status`);
	},

	testDask: async (scheduler: string): Promise<DaskTestResult> => {
		return apiClient.post<DaskTestResult>('/api/v1/simulations/test-dask', { scheduler });
	}
};
