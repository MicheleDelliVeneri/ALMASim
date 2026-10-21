import { fireEvent, render, screen, within } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';

import { toRecords } from '#lib/utils/archive.js';
import ResultsTable from './ResultsTable.svelte';

const COLUMNS = ['proposal_id', 'ALMA_source_name', 'Band', 'member_ous_uid', 'PWV'];
const records = toRecords(COLUMNS, [
	['2023.1.00133.S', 'Cloverleaf', '4', 'uid://A001/X3660/Xe', '1.2'],
	['2023.1.00133.S', 'SDP9', '5', 'uid://A001/X3621/X39cd', '0.8'],
	['2022.1.00001.S', 'M82', '3', 'uid://A001/X2/X1', '2.0']
]);

function table() {
	const props = $state({ records, columns: COLUMNS, selected: new Set<number>() });
	render(ResultsTable, props);
	return props;
}

describe('ResultsTable', () => {
	it('shows the pinned and default columns, hiding the verbose ones', () => {
		table();
		const headers = screen.getAllByRole('columnheader').map((h) => h.textContent?.trim());
		expect(headers).toEqual(['', 'Proposal ID', 'Source Name', 'Band', 'Member OUS UID']);
		expect(headers).not.toContain('PWV');
	});

	it('ticks rows one by one, or every row a filter shows', async () => {
		const props = table();
		await fireEvent.click(screen.getByRole('checkbox', { name: 'Tick row 2' }));
		expect([...props.selected]).toEqual([1]);

		await fireEvent.input(screen.getByRole('textbox', { name: 'Filter Proposal ID' }), {
			target: { value: '2023*' }
		});
		expect(screen.getByText('2 of 3 rows', { exact: false })).toBeInTheDocument();
		await fireEvent.click(screen.getByRole('checkbox', { name: 'Tick every row shown' }));
		expect([...props.selected].sort()).toEqual([0, 1]);

		const body = screen.getAllByRole('rowgroup')[1];
		expect(within(body).queryByText('M82')).toBeNull();
		await fireEvent.click(screen.getByRole('button', { name: 'Clear ticks' }));
		expect(props.selected.size).toBe(0);
	});
});

describe('ResultsTable cells', () => {
	it('rounds long decimals and keeps the full value as the title', () => {
		const props = $state({
			records: toRecords(['proposal_id', 'Ang.res.'], [['P', '2.2268121773845264']]),
			columns: ['proposal_id', 'Ang.res.'],
			selected: new Set<number>()
		});
		render(ResultsTable, props);
		const cell = screen.getByText('2.22681');
		expect(cell).toHaveAttribute('title', '2.2268121773845264');
	});
});
