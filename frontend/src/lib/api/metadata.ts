/** Metadata API types and functions */
import { apiClient } from './client';

export interface MetadataQuery {
	source_name?: string;
	science_keyword?: string[];
	scientific_category?: string[];
	bands?: number[];
	antenna_arrays?: string;
	angular_resolution_range?: [number, number];
	observation_date_range?: [string, string];
	qa2_status?: string[];
	obs_type?: string;
	fov_range?: [number, number];
	time_resolution_range?: [number, number];
	frequency_range?: [number, number];
}

export interface MetadataResponse {
	count: number;
	data: Record<string, unknown>[];
}

export interface ScienceTypes {
	keywords: string[];
	categories: string[];
}

export interface MetadataQueryStartResponse {
	query_id: string;
	status: string;
}

export interface MetadataPageResponse {
	query_id: string;
	page: number;
	rows: Record<string, unknown>[];
	page_size: number;
	total_fetched: number;
	done: boolean;
	error?: string;
}

export interface MetadataSaveRequest {
	path: string;
	data: Record<string, unknown>[];
}

export interface MetadataSaveResponse {
	path: string;
	count: number;
	message?: string;
}

const SCIENCE_TYPES_CACHE_KEY = 'almasim:science-types';
const SCIENCE_TYPES_CACHE_TTL = 1000 * 60 * 60 * 6; // 6 hours

interface CachedScienceTypes {
	timestamp: number;
	data: ScienceTypes;
}

let scienceTypesMemoryCache: ScienceTypes | null = null;
let scienceTypesPromise: Promise<ScienceTypes> | null = null;

const getPersistedScienceTypes = (): ScienceTypes | null => {
	if (typeof window === 'undefined') return scienceTypesMemoryCache;
	try {
		const raw = window.localStorage.getItem(SCIENCE_TYPES_CACHE_KEY);
		if (!raw) return null;
		const parsed = JSON.parse(raw) as CachedScienceTypes;
		if (Date.now() - parsed.timestamp > SCIENCE_TYPES_CACHE_TTL) return null;
		return parsed.data;
	} catch {
		return null;
	}
};

const persistScienceTypes = (data: ScienceTypes) => {
	scienceTypesMemoryCache = data;
	if (typeof window === 'undefined') return;
	try {
		const payload: CachedScienceTypes = { timestamp: Date.now(), data };
		window.localStorage.setItem(SCIENCE_TYPES_CACHE_KEY, JSON.stringify(payload));
	} catch {
		/* ignore storage failures */
	}
};

const fetchScienceTypes = async (): Promise<ScienceTypes> => {
	const data = await apiClient.get<ScienceTypes>('/api/v1/metadata/science-types');
	persistScienceTypes(data);
	return data;
};

export const metadataApi = {
	getScienceTypes: async (): Promise<ScienceTypes> => {
		const cached = scienceTypesMemoryCache || getPersistedScienceTypes();
		if (cached) {
			scienceTypesMemoryCache = cached;
			return cached;
		}
		if (!scienceTypesPromise) {
			scienceTypesPromise = fetchScienceTypes().finally(() => {
				scienceTypesPromise = null;
			});
		}
		return scienceTypesPromise;
	},

	query: async (query: MetadataQuery): Promise<MetadataResponse> => {
		return apiClient.post<MetadataResponse>('/api/v1/metadata/query', query);
	},

	startQuery: async (query: MetadataQuery): Promise<MetadataQueryStartResponse> => {
		return apiClient.post<MetadataQueryStartResponse>('/api/v1/metadata/query/start', query);
	},

	getQueryPage: async (
		queryId: string,
		page: number,
		pageSize = 500
	): Promise<MetadataPageResponse> => {
		return apiClient.get<MetadataPageResponse>(
			`/api/v1/metadata/query/${queryId}/results?page=${page}&page_size=${pageSize}`
		);
	},

	load: async (filePath: string): Promise<MetadataResponse> => {
		return apiClient.get<MetadataResponse>(`/api/v1/metadata/load/${filePath}`);
	},

	save: async (payload: MetadataSaveRequest): Promise<MetadataSaveResponse> => {
		return apiClient.post<MetadataSaveResponse>('/api/v1/metadata/save', payload);
	}
};
