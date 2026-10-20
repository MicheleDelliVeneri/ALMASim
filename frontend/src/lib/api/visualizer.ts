/** Visualizer API: product files in a directory and their integrated images. */
import { apiClient, type RequestOptions } from './client';

export interface DatacubeFile {
	name: string;
	path: string;
	size: number;
	modified: number;
	type?: string;
}

export interface FileListResponse {
	files: DatacubeFile[];
	output_dir: string;
}

export interface IntegratedImage {
	image: number[][];
	stats: {
		shape: number[];
		window_shape: number[];
		integrated_shape: number[];
		min: number;
		max: number;
		mean: number;
		std: number;
		cube_name: string;
		format: string;
		integration_axis: number;
		complex_component: string;
		channel_start: number;
		channel_end: number;
	};
	method: string;
}

/** A product to integrate: an uploaded file, or a file on the server. */
export type IntegrateSource = { file: File } | { serverPath: string; dir?: string };

export interface IntegrateParams {
	method: 'sum' | 'mean';
	complexComponent?: string;
	integrationAxis?: string;
	channelStart?: string;
	channelEnd?: string;
}

export const visualizerApi = {
	listFiles(dir: string, options?: RequestOptions): Promise<FileListResponse> {
		return apiClient.get<FileListResponse>(
			`/api/v1/visualizer/files?dir=${encodeURIComponent(dir)}`,
			options
		);
	},

	integrate(
		source: IntegrateSource,
		params: IntegrateParams,
		options?: RequestOptions
	): Promise<IntegratedImage> {
		const form = new FormData();
		if ('file' in source) {
			form.append('file', source.file);
		} else {
			form.append('server_path', source.serverPath);
			if (source.dir) form.append('dir', source.dir);
		}
		form.append('method', params.method);
		if (params.complexComponent) form.append('complex_component', params.complexComponent);
		if (params.integrationAxis) form.append('integration_axis', params.integrationAxis);
		if (params.channelStart?.trim()) form.append('channel_start', params.channelStart.trim());
		if (params.channelEnd?.trim()) form.append('channel_end', params.channelEnd.trim());
		return apiClient.postForm<IntegratedImage>('/api/v1/visualizer/integrate', form, options);
	}
};
