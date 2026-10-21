/** Simulation API types and functions */
import type { components } from './schema';
import { apiClient, type RequestOptions } from './client';

// Generated from the backend's OpenAPI schema (npm run gen:api, MADIV-28 T7.4).
type Schemas = components['schemas'];

export type ObservationConfigInput = {
	name: string;
	array_type: '12m' | '7m' | 'TP';
	antenna_array: string;
	total_time_s: number;
	correlator?: string;
};

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
	line_sens_10kms?: number;
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
	ms_save_mode?: string;
	inject_serendipitous?: boolean;
	robust?: number;
	imaging_algorithm?: 'legacy' | 'ducc0';
	compute_backend?: string;
	compute_backend_config?: Record<string, unknown>;
	ground_temperature_k?: number;
	correlator?: string;
	elevation_deg?: number;
	source_offset_x_arcsec?: number;
	source_offset_y_arcsec?: number;
	background_mode?: string;
	background_level?: number;
	background_seed?: number;
}

// A request may leave out the fields the server has defaults for.
export type SimulationParamsCreate = Partial<Schemas['SimulationParamsCreate']>;

export type SimulationResponse = Schemas['SimulationResponse'];

export type SimulationStatus = Schemas['SimulationStatus'];

export type SimulationSummary = Schemas['SimulationSummary'];

export type SimulationEstimate = Schemas['SimulationEstimate'];

export interface DaskTestResult {
	ok: boolean;
	scheduler: string;
	dashboard_port: number;
	workers: number;
	total_threads: number;
	total_memory_gb: number;
	error?: string | null;
}

/**
 * One status-socket message. The server sends `{"error": ...}` instead of a
 * status for a simulation it does not know (for example after a restart);
 * that becomes an Error rather than a status without a `status` field.
 */
export function parseStatusMessage(raw: string): SimulationStatus {
	const data = JSON.parse(raw) as Partial<SimulationStatus> & { error?: unknown };
	if (typeof data.status !== 'string') {
		const reason = typeof data.error === 'string' ? data.error : 'malformed status message';
		throw new Error(`Status unavailable: ${reason}`);
	}
	return data as SimulationStatus;
}

export const simulationApi = {
	create: async (params: SimulationParamsCreate): Promise<SimulationResponse> => {
		return apiClient.post<SimulationResponse>('/api/v1/simulations/', params);
	},

	cancel: async (simulationId: string): Promise<SimulationResponse> => {
		return apiClient.post<SimulationResponse>(`/api/v1/simulations/${simulationId}/cancel`, {});
	},

	estimate: async (
		params: SimulationParamsCreate,
		options?: RequestOptions
	): Promise<SimulationEstimate> => {
		return apiClient.post<SimulationEstimate>('/api/v1/simulations/estimate', params, options);
	},

	getStatus: async (simulationId: string): Promise<SimulationStatus> => {
		return apiClient.get<SimulationStatus>(`/api/v1/simulations/${simulationId}/status`);
	},

	list: async (options?: RequestOptions): Promise<SimulationSummary[]> => {
		const data = await apiClient.get<{ simulations?: SimulationSummary[] }>(
			'/api/v1/simulations/',
			options
		);
		return data.simulations ?? [];
	},

	/** Live status of a simulation; the socket closes when the simulation ends. */
	openStatusSocket: (
		simulationId: string,
		onStatus: (status: SimulationStatus) => void,
		onError?: (error: unknown) => void
	): WebSocket => {
		const socket = new WebSocket(apiClient.socketUrl(`/api/v1/simulations/${simulationId}/ws`));
		socket.onmessage = (event) => {
			try {
				onStatus(parseStatusMessage(event.data));
			} catch (error) {
				onError?.(error);
				socket.close();
			}
		};
		return socket;
	},

	testDask: async (scheduler: string): Promise<DaskTestResult> => {
		return apiClient.post<DaskTestResult>('/api/v1/simulations/test-dask', { scheduler });
	}
};
