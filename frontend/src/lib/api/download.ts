/** Download API types and functions */
import { apiClient } from './client';

export interface DataProductInfo {
	access_url: string;
	uid: string;
	filename: string;
	content_length: number;
	content_type: string;
	product_type: string;
	size_mb: number;
}

export interface ResolveProductsResponse {
	products: DataProductInfo[];
	total_count: number;
	total_size_bytes: number;
	total_size_display: string;
	by_type: Record<string, { count: number; size_bytes: number; size_display: string }>;
}

export interface DiskSpaceInfo {
	path: string;
	total_bytes: number;
	used_bytes: number;
	free_bytes: number;
	total_display: string;
	used_display: string;
	free_display: string;
	sufficient: boolean;
}

export interface StartDownloadResponse {
	job_id: string;
	status: string;
	total_files: number;
	total_bytes: number;
	total_size_display: string;
	destination: string;
}

export interface FileStatus {
	filename: string;
	content_length: number;
	bytes_downloaded: number;
	status: string;
	error: string | null;
	progress: number;
}

export interface DownloadJobStatus {
	job_id: string;
	status: string;
	destination: string;
	total_files: number;
	files_completed: number;
	files_failed: number;
	total_bytes: number;
	bytes_downloaded: number;
	progress: number;
	error: string | null;
	files: FileStatus[];
}

export interface DownloadJobSummary {
	job_id: string;
	status: string;
	destination: string;
	total_files: number;
	files_completed: number;
	files_failed: number;
	progress: number;
	created_at: string;
	member_ous_uids: string[];
	product_filter: string;
	total_bytes: number;
	bytes_downloaded: number;
	error: string | null;
}

export interface BrowseDirectoryEntry {
	name: string;
	path: string;
	is_dir: boolean;
}

export interface BrowseDirectoryResponse {
	current: string;
	parent: string | null;
	entries: BrowseDirectoryEntry[];
}

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
	}): Promise<StartDownloadResponse> {
		return apiClient.post<StartDownloadResponse>('/api/v1/downloads/start', {
			member_ous_uids: params.memberOusUids,
			product_filter: params.productFilter ?? 'all',
			destination: params.destination,
			max_parallel: params.maxParallel ?? 3,
			extract_tar: params.extractTar ?? false
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

	/** Cancel a download job */
	cancelJob(jobId: string): Promise<{ message: string }> {
		return apiClient.post<{ message: string }>(
			`/api/v1/downloads/jobs/${encodeURIComponent(jobId)}/cancel`
		);
	},

	/** Re-download a previous job using stored file records */
	redownloadJob(jobId: string): Promise<StartDownloadResponse> {
		return apiClient.post<StartDownloadResponse>(
			`/api/v1/downloads/jobs/${encodeURIComponent(jobId)}/redownload`
		);
	},

	/** Delete a download job from history */
	deleteJob(jobId: string): Promise<{ job_id: string; deleted: boolean }> {
		return apiClient.delete<{ job_id: string; deleted: boolean }>(
			`/api/v1/downloads/jobs/${encodeURIComponent(jobId)}`
		);
	}
};
