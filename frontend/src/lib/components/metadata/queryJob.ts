/** A background metadata query the page polls (MADIV-28 T7.5). */
import type { MetadataQuery, MetadataRow } from '../../api/metadata';

export type QueryJob = {
	id: string;
	status: 'starting' | 'fetching' | 'done' | 'cancelled' | 'error';
	rows: MetadataRow[];
	message: string;
	error: string;
	pollTimer: ReturnType<typeof setTimeout> | null;
	page: number;
	query: MetadataQuery;
};
