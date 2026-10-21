/** Metadata API types and functions */
import type { components } from './schema';
import { apiClient } from './client';
import { encodePathSegments } from '#lib/utils/paths.js';

// Generated from the backend's OpenAPI schema (npm run gen:api, MADIV-28 T7.4).
type Schemas = components['schemas'];

// A request may leave out every field; the server fills in the defaults.
export type MetadataQuery = Partial<Schemas['MetadataQuery']>;

/**
 * One metadata row: the archive columns the backend returns
 * (``almasim.services.metadata.tap.queries.ALL_COLUMNS``), plus any others.
 */
export type MetadataRow = {
	ALMA_source_name?: string;
	Band?: number | string;
	Array_type?: string;
	antenna_arrays?: string;
	'Ang.res.'?: number;
	'Obs.date'?: string;
	Project_abstract?: string;
	science_keyword?: string;
	scientific_category?: string;
	QA2_status?: string;
	Type?: string;
	PWV?: number;
	SB_name?: string;
	'Vel.res.'?: number;
	RA?: number;
	Dec?: number;
	FOV?: number;
	'Int.Time'?: number;
	Cont_sens_mJybeam?: number;
	Line_sens_10kms_mJybeam?: number;
	Bandwidth?: number;
	Freq?: number;
	'Freq.sup.'?: string;
	proposal_id?: string;
	member_ous_uid?: string;
	group_ous_uid?: string;
	[column: string]: unknown;
};

export type MetadataResponse = Omit<Schemas['MetadataResponse'], 'data'> & {
	data: MetadataRow[];
};

export interface ScienceTypes {
	keywords: string[];
	categories: string[];
}

export type MetadataQueryStartResponse = Schemas['MetadataQueryStartResponse'];

export type MetadataPageResponse = Omit<Schemas['MetadataPageResponse'], 'rows'> & {
	rows: MetadataRow[];
};

export type MetadataSaveRequest = Schemas['MetadataSaveRequest'];

export type MetadataSaveResponse = Schemas['MetadataSaveResponse'];

export type QueryPresetCreate = Schemas['QueryPresetCreate'];

export interface QueryPreset {
	name: string;
	description: string;
	filters: MetadataQuery;
	result_count: number;
	created_at: string;
}

// The backend stores a preset's filters as a plain object; they are a MetadataQuery.
export interface QueryPresetsResponse {
	presets: QueryPreset[];
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

	cancelQuery: async (queryId: string): Promise<void> => {
		return apiClient.delete(`/api/v1/metadata/query/${queryId}`);
	},

	load: async (filePath: string): Promise<MetadataResponse> => {
		return apiClient.get<MetadataResponse>(`/api/v1/metadata/load/${encodePathSegments(filePath)}`);
	},

	save: async (payload: MetadataSaveRequest): Promise<MetadataSaveResponse> => {
		return apiClient.post<MetadataSaveResponse>('/api/v1/metadata/save', payload);
	},

	listPresets: async (): Promise<QueryPresetsResponse> => {
		return apiClient.get<QueryPresetsResponse>('/api/v1/metadata/presets');
	},

	savePreset: async (payload: QueryPresetCreate): Promise<QueryPreset> => {
		return apiClient.post<QueryPreset>('/api/v1/metadata/presets', payload);
	},

	deletePreset: async (name: string): Promise<void> => {
		return apiClient.delete(`/api/v1/metadata/presets/${encodeURIComponent(name)}`);
	}
};
