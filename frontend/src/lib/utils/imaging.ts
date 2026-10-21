/**
 * The imaging workflow: the per-window rows `image compute-parameters`
 * writes, the task folder `image image-from-ms` makes for each, and each
 * task's progress from its markers (almasim.services.imaging.markers):
 * `<out>/<ms stem>/SPW-<n>[_FIELD-<m>]` with `.done`, `.failed` and `.log`
 * beside it.
 */
import { RUN_DIR } from './runDir';

export const PARAMETERS_CSV = 'imaging_parameters.csv';
export const QUALITY_CSV = 'clean-quality.csv';

export type TaskState = 'done' | 'failed' | 'running' | 'waiting';

export interface Task {
	/** The MS's folder under the output root (its name without the last suffix). */
	msDir: string;
	/** SPW-25, or SPW-25_FIELD-2 when the row images one field. */
	name: string;
	spw: string;
	field: string;
	frequencyGhz: number;
	channels: number;
	beamArcsec: number;
	fovArcsec: number;
	minutes: number;
	source: string;
}

/** `Path(name).stem` as Python has it: the name without its last suffix. */
export function stem(path: string): string {
	const name = path.replace(/\/+$/, '').split('/').pop() ?? '';
	const dot = name.lastIndexOf('.');
	return dot > 0 ? name.slice(0, dot) : name;
}

/** One parameters row as an imaging task (wsclean_plan.task_dir_for_row). */
export function taskOf(row: Record<string, string>, outputRoot: string): Task {
	const fields = (row.target_field_ids ?? '').split(',').filter((f) => f.trim() !== '');
	const spw = String(parseInt(row.spectral_window_id, 10));
	const single = fields.length === 1 ? fields[0].trim() : '';
	const name = single ? `SPW-${spw}_FIELD-${single}` : `SPW-${spw}`;
	return {
		msDir: `${outputRoot.replace(/\/+$/, '')}/${stem(row.filename)}`,
		name,
		spw,
		field: fields.join(','),
		frequencyGhz: Number(row.reference_frequency) / 1e9,
		channels: Number(row.n_channels),
		beamArcsec: Number(row.synthetized_beam_size),
		fovArcsec: Number(row.fov_per_frequency),
		minutes: Number(row.duration_s) / 60,
		source: row.field_name ?? ''
	};
}

/** A short name for an MS folder: its execution block's last part (Xb670 for uid___A002_X113d9b7_Xb670.ms.split). */
export function shortMs(msDir: string): string {
	const name = msDir.split('/').pop() ?? '';
	const base = name.split('.')[0];
	return base.includes('_') ? (base.split('_').pop() ?? base) : base;
}

/** The clean-quality summary's columns worth a glance, in this order, when present. */
export const QUALITY_COLUMNS: Record<string, string> = {
	'continuum.snr': 'S/N',
	'continuum.residual_share_1beam': 'Residual share (1 beam)',
	'continuum.residual_peak_sigma': 'Residual peak σ',
	'continuum.model_flux_jy': 'Model flux (Jy)',
	'cube.line_channels': 'Line channels',
	'cube.channel_noise_mjy': 'Channel noise (mJy)'
};

/** A summary value as shown: numbers to three significant digits. */
export function qualityValue(value: string): string {
	if (value === '' || !Number.isFinite(Number(value))) return value || '—';
	return String(Number(Number(value).toPrecision(3)));
}

/** A task's state from the names in its MS folder. */
export function taskState(task: Task, names: Set<string>): TaskState {
	if (names.has(`${task.name}.done`)) return 'done';
	if (names.has(`${task.name}.failed`)) return 'failed';
	if (names.has(`${task.name}.log`) || names.has(task.name)) return 'running';
	return 'waiting';
}

export function countStates(states: TaskState[]): Record<TaskState, number> {
	const counts: Record<TaskState, number> = { done: 0, failed: 0, running: 0, waiting: 0 };
	for (const s of states) counts[s]++;
	return counts;
}

/** `image compute-parameters` for a folder, its CSV in the run folder. */
export function parametersArgv(folder: string, pattern: string, allSpws: boolean): string[] {
	const argv = ['image', 'compute-parameters', folder, `${RUN_DIR}/${PARAMETERS_CSV}`];
	if (pattern.trim() && pattern.trim() !== '*.cal') argv.push('--pattern', pattern.trim());
	if (allSpws) argv.push('--all-spws');
	return argv;
}

/** `image image-from-ms` over the parameters CSV, with the form's options. */
export function imagingArgv(
	parametersCsv: string,
	outputRoot: string,
	options: string[]
): string[] {
	return ['image', 'image-from-ms', parametersCsv, outputRoot, ...options];
}

/** `image clean-quality` over what imaging made. */
export function qualityArgv(outputRoot: string): string[] {
	return ['image', 'clean-quality', outputRoot];
}
