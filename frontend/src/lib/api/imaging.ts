/** Imaging API types and functions */
import type { components } from './schema';
import { apiClient } from './client';

// Generated from the backend's OpenAPI schema (npm run gen:api, MADIV-28 T7.4).
type Schemas = components['schemas'];

export type ImagePreviewPayload = Schemas['ImagePreviewPayload'];

export type DeconvolutionResponse = Schemas['DeconvolutionResponse'];

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
