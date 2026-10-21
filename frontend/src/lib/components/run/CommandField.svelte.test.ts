import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';

import type { Param } from '#lib/utils/commandForm.js';
import CommandField from './CommandField.svelte';

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

/** Render a field and give back its current value through the bound prop. */
function field(p: Param, value: string | boolean | string[]) {
	const props = $state({ param: p, value });
	render(CommandField, props);
	return props;
}

describe('CommandField', () => {
	it('draws a flag as a labelled checkbox with its help', async () => {
		const props = field(param({ name: 'cube', type: 'boolean', help: 'Image a cube' }), false);
		const box = screen.getByRole('checkbox', { name: /Cube/ });
		expect(screen.getByText('Image a cube')).toBeInTheDocument();
		expect(screen.getByText('--cube')).toBeInTheDocument();
		await fireEvent.click(box);
		expect(props.value).toBe(true);
	});

	it('offers a choice as a select', async () => {
		const props = field(
			param({
				name: 'weighting',
				type: 'choice',
				choices: ['briggs', 'natural'],
				default: 'briggs'
			}),
			'briggs'
		);
		const select = screen.getByRole('combobox', { name: /Weighting/ });
		expect(screen.getAllByRole('option').map((o) => o.textContent)).toEqual(['briggs', 'natural']);
		await fireEvent.change(select, { target: { value: 'natural' } });
		expect(props.value).toBe('natural');
	});

	it('keeps numbers in range, as text', async () => {
		const props = field(param({ name: 'fov_fraction', type: 'number', min: 0, max: 4 }), '1.5');
		const input = screen.getByRole('spinbutton', { name: /Fov fraction/ });
		expect(input).toHaveAttribute('min', '0');
		expect(input).toHaveAttribute('max', '4');
		expect(input).toHaveValue(1.5);
		await fireEvent.input(input, { target: { value: '2' } });
		expect(props.value).toBe('2');
	});

	it('takes a list one value per line', async () => {
		const props = field(param({ name: 'selfcal', multiple: true }), []);
		await fireEvent.input(screen.getByRole('textbox', { name: /Selfcal/ }), {
			target: { value: 'inf\n 30s \n\n' }
		});
		expect(props.value).toEqual(['inf', '30s']);
	});

	it('marks a required argument and names it as the CLI does', () => {
		field(param({ name: 'out', kind: 'argument', flags: ['out'], required: true }), '');
		expect(screen.getByRole('textbox', { name: /Out \*/ })).toBeRequired();
		expect(screen.getByText('OUT')).toBeInTheDocument();
	});

	it('browses for a path', () => {
		field(param({ name: 'params_csv', type: 'path', path_kind: 'file' }), '/data/p.csv');
		expect(screen.getByRole('textbox', { name: /Params csv/ })).toHaveValue('/data/p.csv');
		expect(screen.getByRole('button', { name: 'Browse…' })).toBeInTheDocument();
	});
});
