/** Download API types and functions */
import type { components } from './schema';
import { apiClient } from './client';
import type { MetadataResponse, MetadataRow } from './metadata';

// Generated from the backend's OpenAPI schema (npm run gen:api, MADIV-28 T7.4).
type Schemas = components['schemas'];

export type DataProductInfo = Schemas['DataProductInfo'];

export type ResolveProductsResponse = Schemas['ResolveProductsResponse'];

export type DiskSpaceInfo = Schemas['DiskSpaceInfo'];

export type StartDownloadResponse = Schemas['StartDownloadResponse'];

export type FileStatus = Schemas['FileStatus'];

export type DownloadJobStatus = Schemas['DownloadJobStatus'];

export type DownloadJobSummary = Schemas['DownloadJobSummary'];

export type BrowseDirectoryEntry = Schemas['BrowseDirectoryEntry'];

export type BrowseDirectoryResponse = Schemas['BrowseDirectoryResponse'];

export const downloadApi = {
	/** Browse server-side directories */
	browseDirectory(path: string = '/'): Promise<BrowseDirectoryResponse> {
		return apiClient.get<BrowseDirectoryResponse>(
			`/api/v1/downloads/browse?path=${encodeURIComponent(path)}`
		);
	},

	/** Create a directory and return its browse result */
	createDirectory(path: string): Promise<BrowseDirectoryResponse> {
		return apiClient.post<BrowseDirectoryResponse>(
			`/api/v1/downloads/mkdir?path=${encodeURIComponent(path)}`
		);
	},

	/** Resolve downloadable products for selected member_ous_uids */
	resolveProducts(memberOusUids: string[]): Promise<ResolveProductsResponse> {
		return apiClient.post<ResolveProductsResponse>('/api/v1/downloads/resolve', {
			member_ous_uids: memberOusUids
		});
	},

	/** Check disk space at a given path */
	checkDiskSpace(path: string, neededBytes: number = 0): Promise<DiskSpaceInfo> {
		return apiClient.post<DiskSpaceInfo>('/api/v1/downloads/disk-space', {
			path,
			needed_bytes: neededBytes
		});
	},

	/** Start a download job */
	startDownload(params: {
		memberOusUids: string[];
		productFilter?: string;
		destination: string;
		maxParallel?: number;
		extractTar?: boolean;
		unpackMs?: boolean;
		generateCalibratedVisibilities?: boolean;
		cleanIntermediateFiles?: boolean;
		archiveOutputRoot?: string;
		casaDataRoot?: string;
		skipCasaDataUpdate?: boolean;
		selectedMetadata?: MetadataRow[];
	}): Promise<StartDownloadResponse> {
		return apiClient.post<StartDownloadResponse>('/api/v1/downloads/start', {
			member_ous_uids: params.memberOusUids,
			product_filter: params.productFilter ?? 'all',
			destination: params.destination,
			max_parallel: params.maxParallel ?? 3,
			extract_tar: params.extractTar ?? false,
			unpack_ms: params.unpackMs ?? false,
			generate_calibrated_visibilities: params.generateCalibratedVisibilities ?? false,
			clean_intermediate_files: params.cleanIntermediateFiles ?? false,
			archive_output_root: params.archiveOutputRoot || null,
			casa_data_root: params.casaDataRoot || null,
			skip_casa_data_update: params.skipCasaDataUpdate ?? false,
			selected_metadata: params.selectedMetadata ?? []
		});
	},

	/** List all download jobs */
	listJobs(): Promise<DownloadJobSummary[]> {
		return apiClient.get<DownloadJobSummary[]>('/api/v1/downloads/jobs');
	},

	/** Get detailed status for a specific job */
	getJobStatus(jobId: string): Promise<DownloadJobStatus> {
		return apiClient.get<DownloadJobStatus>(`/api/v1/downloads/jobs/${encodeURIComponent(jobId)}`);
	},

	/** Load metadata rows associated with a downloaded data product */
	loadJobMetadata(jobId: string): Promise<MetadataResponse> {
		return apiClient.get<MetadataResponse>(
			`/api/v1/downloads/jobs/${encodeURIComponent(jobId)}/metadata`
		);
	},

	/** Cancel a download job */
	cancelJob(jobId: string): Promise<{ message: string }> {
		return apiClient.post<{ message: string }>(
			`/api/v1/downloads/jobs/${encodeURIComponent(jobId)}/cancel`
		);
	},

	/** Re-download a previous job using stored file records */
	redownloadJob(
		jobId: string,
		params: {
			maxParallel?: number;
			extractTar?: boolean;
			unpackMs?: boolean;
			generateCalibratedVisibilities?: boolean;
			cleanIntermediateFiles?: boolean;
			archiveOutputRoot?: string;
			casaDataRoot?: string;
			skipCasaDataUpdate?: boolean;
		} = {}
	): Promise<StartDownloadResponse> {
		return apiClient.post<StartDownloadResponse>(
			`/api/v1/downloads/jobs/${encodeURIComponent(jobId)}/redownload`,
			{
				max_parallel: params.maxParallel,
				extract_tar: params.extractTar,
				unpack_ms: params.unpackMs,
				generate_calibrated_visibilities: params.generateCalibratedVisibilities,
				clean_intermediate_files: params.cleanIntermediateFiles,
				archive_output_root: params.archiveOutputRoot || null,
				casa_data_root: params.casaDataRoot || null,
				skip_casa_data_update: params.skipCasaDataUpdate
			}
		);
	},

	/** Delete a download job from history */
	deleteJob(jobId: string): Promise<{ job_id: string; deleted: boolean }> {
		return apiClient.delete<{ job_id: string; deleted: boolean }>(
			`/api/v1/downloads/jobs/${encodeURIComponent(jobId)}`
		);
	}
};
