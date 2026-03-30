/** Imaging API types and functions */
import { apiClient } from './client';

export interface ImagePreviewPayload {
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

export interface DeconvolutionResponse {
	dirty: ImagePreviewPayload;
	component_model: ImagePreviewPayload;
	restored: ImagePreviewPayload;
	residual: ImagePreviewPayload;
	reference_clean: ImagePreviewPayload | null;
	convolved_reference: ImagePreviewPayload | null;
	metadata: {
		dirty_cube_path: string;
		beam_cube_path: string;
		beam_cube_name: string;
		clean_cube_path: string | null;
		cycles_requested: number;
		cycles_completed: number;
		cycles_added: number;
		gain: number;
		threshold: number | null;
		state_path: string;
		component_cube_path: string;
		restored_cube_path: string;
		residual_cube_path: string;
		convolved_reference_path: string | null;
		resumed: boolean;
	};
}

export const imagingApi = {
	deconvolve(params: {
		directory: string;
		dirtyCubePath: string;
		beamCubePath: string;
		cleanCubePath?: string | null;
		cycles?: number;
		gain?: number;
		threshold?: number | null;
		statePath?: string | null;
		method?: 'sum' | 'mean';
	}): Promise<DeconvolutionResponse> {
		return apiClient.post<DeconvolutionResponse>('/api/v1/imaging/deconvolve', {
			directory: params.directory,
			dirty_cube_path: params.dirtyCubePath,
			beam_cube_path: params.beamCubePath,
			clean_cube_path: params.cleanCubePath ?? null,
			cycles: params.cycles ?? 100,
			gain: params.gain ?? 0.1,
			threshold: params.threshold ?? null,
			state_path: params.statePath ?? null,
			method: params.method ?? 'sum'
		});
	}
};
