/** Simulation API types and functions */
import { apiClient } from './client';

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

export const simulationApi = {
	create: async (params: SimulationParamsCreate): Promise<SimulationResponse> => {
		return apiClient.post<SimulationResponse>('/api/v1/simulations/', params);
	},

	getStatus: async (simulationId: string): Promise<SimulationStatus> => {
		return apiClient.get<SimulationStatus>(`/api/v1/simulations/${simulationId}/status`);
	}
};
