import { describe, expect, it } from 'vitest';

import {
	buildArgv,
	initialValues,
	parseArgv,
	searchCommands,
	sections,
	type Command,
	type Param,
	type Values
} from './commandForm';

const param = (p: Partial<Param> & Pick<Param, 'name'>): Param => ({
	kind: 'option',
	flags: [`--${p.name.replace(/_/g, '-')}`],
	negative_flags: [],
	required: false,
	multiple: false,
	default: null,
	help: '',
	panel: null,
	type: 'string',
	...p
});

const IMAGE: Command = {
	name: 'image image-from-ms',
	summary: 'Image calibrated measurement sets with WSClean',
	help: '',
	executor: 'slurm',
	needs_internet: false,
	extra_args: false,
	params: [
		param({
			name: 'params_csv',
			kind: 'argument',
			flags: ['params_csv'],
			required: true,
			type: 'path',
			path_kind: 'file'
		}),
		param({
			name: 'out',
			kind: 'argument',
			flags: ['out'],
			required: true,
			type: 'path',
			path_kind: 'folder'
		}),
		param({ name: 'fov_fraction', type: 'number', default: 1.5, panel: 'Imaging', min: 0 }),
		param({ name: 'cube', type: 'boolean', default: false, panel: 'Spectral cube' }),
		param({
			name: 'multiscale',
			type: 'boolean',
			default: true,
			negative_flags: ['--no-multiscale'],
			panel: 'Imaging'
		}),
		param({
			name: 'weighting',
			type: 'choice',
			choices: ['briggs', 'natural'],
			default: 'briggs',
			panel: 'Imaging'
		}),
		param({ name: 'selfcal', multiple: true, panel: 'Self-calibration' }),
		param({
			name: 'backend',
			type: 'choice',
			choices: ['sync', 'slurm'],
			default: 'sync',
			panel: 'Cluster'
		}),
		param({ name: 'slurm_queue', default: 'normal', panel: 'Cluster' })
	]
};

describe('initialValues', () => {
	it('starts every field at its default', () => {
		expect(initialValues(IMAGE)).toEqual({
			params_csv: '',
			out: '',
			fov_fraction: '1.5',
			cube: false,
			multiscale: true,
			weighting: 'briggs',
			selfcal: [],
			backend: 'sync',
			slurm_queue: 'normal'
		});
	});
});

describe('buildArgv', () => {
	it('gives the arguments, then only the options that changed', () => {
		const values: Values = { ...initialValues(IMAGE), params_csv: 'p.csv', out: '/data/out' };
		expect(buildArgv(IMAGE, values)).toEqual(['image', 'image-from-ms', 'p.csv', '/data/out']);
		values.fov_fraction = '2';
		values.cube = true;
		values.multiscale = false;
		values.selfcal = ['inf', '30s'];
		expect(buildArgv(IMAGE, values)).toEqual([
			'image', 'image-from-ms', 'p.csv', '/data/out',
			'--fov-fraction', '2',
			'--cube',
			'--no-multiscale',
			'--selfcal', 'inf', '--selfcal', '30s'
		]); // prettier-ignore
	});

	it('leaves out empty fields and adds extra arguments last', () => {
		const values = { ...initialValues(IMAGE), params_csv: 'p.csv', fov_fraction: '' };
		expect(buildArgv(IMAGE, values, ['--', '-niter', '1000'])).toEqual([
			'image', 'image-from-ms', 'p.csv', '--', '-niter', '1000'
		]); // prettier-ignore
	});
});

describe('parseArgv', () => {
	it('turns a submitted command line back into the form', () => {
		const argv = [
			'image', 'image-from-ms', 'p.csv', '/data/out',
			'--fov-fraction=2', '--cube', '--no-multiscale',
			'--selfcal', 'inf', '--selfcal', '30s',
			'--backend', 'slurm', '--slurm-queue', 'zwickyA-CPU'
		]; // prettier-ignore
		const { values, extra } = parseArgv(IMAGE, argv);
		expect(extra).toEqual([]);
		expect(values).toMatchObject({
			params_csv: 'p.csv',
			out: '/data/out',
			fov_fraction: '2',
			cube: true,
			multiscale: false,
			selfcal: ['inf', '30s'],
			backend: 'slurm',
			slurm_queue: 'zwickyA-CPU'
		});
		// And back again, the same line.
		expect(buildArgv(IMAGE, values, extra)).toEqual([
			'image', 'image-from-ms', 'p.csv', '/data/out',
			'--fov-fraction', '2', '--cube', '--no-multiscale',
			'--selfcal', 'inf', '--selfcal', '30s',
			'--backend', 'slurm', '--slurm-queue', 'zwickyA-CPU'
		]); // prettier-ignore
	});

	it('keeps what the form has no field for', () => {
		const { values, extra } = parseArgv(IMAGE, [
			'image',
			'image-from-ms',
			'a',
			'b',
			'c',
			'--nope',
			'--',
			'-x'
		]);
		expect(values.params_csv).toBe('a');
		expect(extra).toEqual(['c', '--nope', '--', '-x']);
	});
});

describe('sections', () => {
	it('puts the inputs first and folds Cluster and Advanced at the end', () => {
		const out = sections(IMAGE);
		expect(out.map((s) => [s.title, s.folded, s.params.length])).toEqual([
			['Inputs', false, 2],
			['Imaging', false, 3],
			['Spectral cube', false, 1],
			['Self-calibration', false, 1],
			['Cluster', true, 2]
		]);
	});

	it('gathers options without a panel, and never folds a required one', () => {
		const command: Command = {
			...IMAGE,
			params: [param({ name: 'a' }), param({ name: 'b', panel: 'Advanced', required: true })]
		};
		expect(sections(command).map((s) => [s.title, s.folded])).toEqual([
			['Options', false],
			['Advanced', false]
		]);
	});
});

describe('searchCommands', () => {
	it('matches every word in the name or summary', () => {
		const other: Command = { ...IMAGE, name: 'metadata', summary: 'Query the ALMA archive' };
		expect(searchCommands([IMAGE, other], 'image wsclean').map((c) => c.name)).toEqual([
			'image image-from-ms'
		]);
		expect(searchCommands([IMAGE, other], '  ')).toHaveLength(2);
		expect(searchCommands([IMAGE, other], 'archive').map((c) => c.name)).toEqual(['metadata']);
	});
});
