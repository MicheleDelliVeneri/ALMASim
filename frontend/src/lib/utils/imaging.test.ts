import { describe, expect, it } from 'vitest';

import {
	countStates,
	imagingArgv,
	parametersArgv,
	qualityArgv,
	qualityValue,
	shortMs,
	stem,
	taskOf,
	taskState
} from './imaging';

const ROW = {
	filename: '/data/cal/uid___A002_X113d9b7_Xb670.ms.split.cal',
	spectral_window_id: '25',
	reference_frequency: '95601320555.43254',
	fov_per_frequency: '62.03230231926096',
	synthetized_beam_size: '2.574868016340855',
	n_channels: '240',
	duration_s: '3169.7350006103516',
	target_field_ids: '2',
	field_name: 'H-ATLAS_J113526.3014605_G12v2.43'
};

describe('tasks from the parameters rows', () => {
	it('names the MS folder as Python does', () => {
		expect(stem('/data/cal/uid___A002_X1.ms.split.cal')).toBe('uid___A002_X1.ms.split');
		expect(stem('plain')).toBe('plain');
		expect(stem('/data/x.ms/')).toBe('x');
	});

	it('gives each row its task folder and figures', () => {
		const task = taskOf(ROW, '/out/');
		expect(task.msDir).toBe('/out/uid___A002_X113d9b7_Xb670.ms.split');
		expect(task.name).toBe('SPW-25_FIELD-2');
		expect(task.frequencyGhz).toBeCloseTo(95.6013, 4);
		expect(task.channels).toBe(240);
		expect(task.minutes).toBeCloseTo(52.83, 2);
		// Several fields (or none) share one SPW-<n> task.
		expect(taskOf({ ...ROW, target_field_ids: '2,5' }, '/out').name).toBe('SPW-25');
		expect(taskOf({ ...ROW, target_field_ids: '' }, '/out').name).toBe('SPW-25');
	});

	it('reads progress from the markers', () => {
		const task = taskOf(ROW, '/out');
		const state = (...names: string[]) => taskState(task, new Set(names));
		expect(state()).toBe('waiting');
		expect(state('SPW-25_FIELD-2')).toBe('running');
		expect(state('SPW-25_FIELD-2', 'SPW-25_FIELD-2.log')).toBe('running');
		expect(state('SPW-25_FIELD-2', 'SPW-25_FIELD-2.done')).toBe('done');
		expect(state('SPW-25_FIELD-2.failed', 'SPW-25_FIELD-2.log')).toBe('failed');
		expect(state('SPW-27_FIELD-2.done')).toBe('waiting');
		expect(countStates(['done', 'done', 'failed', 'waiting'])).toEqual({
			done: 2,
			failed: 1,
			running: 0,
			waiting: 1
		});
	});
});

describe('showing the results', () => {
	it('names an MS by its execution block', () => {
		expect(shortMs('/out/uid___A002_X113d9b7_Xb670.ms.split')).toBe('Xb670');
		expect(shortMs('/out/plain')).toBe('plain');
	});

	it('rounds the quality figures', () => {
		expect(qualityValue('42.353924511367275')).toBe('42.4');
		expect(qualityValue('0.0008024143080547219')).toBe('0.000802');
		expect(qualityValue('')).toBe('—');
		expect(qualityValue('n/a')).toBe('n/a');
	});
});

describe('the imaging runs', () => {
	it('computes the parameters into the run folder', () => {
		expect(parametersArgv('/data/cal', '*.cal', false)).toEqual([
			'image', 'compute-parameters', '/data/cal', '{run_dir}/imaging_parameters.csv'
		]); // prettier-ignore
		expect(parametersArgv('/data/cal', '*.ms', true).slice(4)).toEqual([
			'--pattern',
			'*.ms',
			'--all-spws'
		]);
	});

	it('images, then measures the clean', () => {
		expect(imagingArgv('/runs/P/imaging_parameters.csv', '/data/images', ['--cube'])).toEqual([
			'image', 'image-from-ms', '/runs/P/imaging_parameters.csv', '/data/images', '--cube'
		]); // prettier-ignore
		expect(qualityArgv('/data/images')).toEqual(['image', 'clean-quality', '/data/images']);
	});
});
