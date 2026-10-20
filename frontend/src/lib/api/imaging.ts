/** Imaging API types and functions */
import type { components } from './schema';
import { apiClient } from './client';

// Generated from the backend's OpenAPI schema (npm run gen:api, MADIV-28 T7.4).
type Schemas = components['schemas'];

export type ImagePreviewPayload = Schemas['ImagePreviewPayload'];

export type WienerPreviewResponse = Schemas['WienerPreviewResponse'];

export const imagingApi = {
	/** A Wiener preview of saved dirty/beam cubes (MADIV-28 T5.12). */
	preview(params: {
		directory: string;
		dirtyCubePath: string;
		beamCubePath: string;
		cleanCubePath?: string | null;
		epsilon?: number;
		method?: 'sum' | 'mean';
	}): Promise<WienerPreviewResponse> {
		return apiClient.post<WienerPreviewResponse>('/api/v1/imaging/preview', {
			directory: params.directory,
			dirty_cube_path: params.dirtyCubePath,
			beam_cube_path: params.beamCubePath,
			clean_cube_path: params.cleanCubePath ?? null,
			epsilon: params.epsilon ?? 1e-3,
			method: params.method ?? 'sum'
		});
	}
};
