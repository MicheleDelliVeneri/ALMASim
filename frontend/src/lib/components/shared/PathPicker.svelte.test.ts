import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';

import PathPicker from './PathPicker.svelte';

const ROOTS = {
	roots: [
		{ path: '/home/astro', exists: true, runs: false },
		{ path: '/data/runs', exists: true, runs: true }
	]
};

const LISTINGS: Record<string, unknown> = {
	'/home/astro': {
		path: '/home/astro',
		parent: '/home',
		truncated: false,
		entries: [
			{ name: 'cubes', kind: 'folder', size: null, modified: 1 },
			{ name: 'notes.txt', kind: 'file', size: 10, modified: 1 }
		]
	},
	'/home/astro/cubes': {
		path: '/home/astro/cubes',
		parent: '/home/astro',
		truncated: false,
		entries: [{ name: 'cube.fits', kind: 'file', size: 99, modified: 1 }]
	}
};

/** A fake console server: the roots, the two listings above, and mkdir. */
function fakeServer() {
	const calls: string[] = [];
	const fetchMock = vi.fn(async (input: Request) => {
		const url = new URL(input.url);
		calls.push(`${input.method} ${url.pathname}${url.search}`);
		const json = (body: unknown, status = 200) =>
			new Response(JSON.stringify(body), {
				status,
				headers: { 'Content-Type': 'application/json' }
			});
		if (url.pathname === '/api/v2/files/roots') return json(ROOTS);
		if (url.pathname === '/api/v2/files/list') {
			const listing = LISTINGS[url.searchParams.get('path') ?? ''];
			return listing ? json(listing) : json({ detail: 'outside the folders' }, 403);
		}
		if (url.pathname === '/api/v2/files/mkdir') {
			const { path } = await input.json();
			LISTINGS[path] = { path, parent: '/home/astro', truncated: false, entries: [] };
			return json({ path }, 201);
		}
		return json({ detail: 'not found' }, 404);
	});
	vi.stubGlobal('fetch', fetchMock);
	return calls;
}

describe('PathPicker', () => {
	afterEach(() => vi.unstubAllGlobals());

	it('is a labelled text field', () => {
		render(PathPicker, { value: '/home/astro/out', label: 'Output folder' });
		expect(screen.getByLabelText('Output folder')).toHaveValue('/home/astro/out');
	});

	it('browses from the first root and chooses a folder', async () => {
		fakeServer();
		render(PathPicker, { value: '', label: 'Output folder' });
		await fireEvent.click(screen.getByRole('button', { name: 'Browse…' }));

		await screen.findByRole('button', { name: /cubes/ });
		// Files are shown, but only folders can be chosen here.
		expect(screen.queryByRole('button', { name: /notes\.txt/ })).toBeNull();
		expect(screen.getByText('notes.txt')).toBeInTheDocument();
		expect(screen.getByRole('button', { name: 'Runs' })).toBeInTheDocument();

		await fireEvent.click(screen.getByRole('button', { name: /cubes/ }));
		await screen.findByText('/home/astro/cubes');
		await fireEvent.click(screen.getByRole('button', { name: 'Choose this folder' }));

		expect(screen.getByLabelText('Output folder')).toHaveValue('/home/astro/cubes');
		expect(screen.queryByRole('dialog')).toBeNull();
	});

	it('chooses a file when a file is wanted', async () => {
		fakeServer();
		render(PathPicker, { value: '/home/astro/cubes/old.fits', label: 'Cube', kind: 'file' });
		await fireEvent.click(screen.getByRole('button', { name: 'Browse…' }));

		// It opens in the folder of the current value.
		const file = await screen.findByRole('button', { name: /cube\.fits/ });
		const choose = screen.getByRole('button', { name: 'Choose file' });
		expect(choose).toBeDisabled();
		await fireEvent.click(file);
		expect(file).toHaveAttribute('aria-pressed', 'true');
		await fireEvent.click(choose);
		expect(screen.getByLabelText('Cube')).toHaveValue('/home/astro/cubes/cube.fits');
	});

	it('falls back to a root when the typed path is outside them', async () => {
		const calls = fakeServer();
		render(PathPicker, { value: '/etc', label: 'Output folder' });
		await fireEvent.click(screen.getByRole('button', { name: 'Browse…' }));
		await screen.findByRole('button', { name: /cubes/ });
		expect(calls).toContain('GET /api/v2/files/list?path=%2Fetc');
		expect(calls).toContain('GET /api/v2/files/list?path=%2Fhome%2Fastro');
	});

	it('creates a folder and opens it', async () => {
		const calls = fakeServer();
		render(PathPicker, { value: '', label: 'Output folder', allowCreate: true });
		await fireEvent.click(screen.getByRole('button', { name: 'Browse…' }));
		await screen.findByRole('button', { name: /cubes/ });

		await fireEvent.input(screen.getByLabelText('New folder name'), {
			target: { value: 'run-42' }
		});
		await fireEvent.click(screen.getByRole('button', { name: 'Create' }));
		await waitFor(() => expect(calls).toContain('POST /api/v2/files/mkdir'));
		await screen.findByText('/home/astro/run-42');
	});

	it('shows why a folder cannot be opened', async () => {
		fakeServer();
		render(PathPicker, { value: '', label: 'Output folder' });
		await fireEvent.click(screen.getByRole('button', { name: 'Browse…' }));
		await screen.findByRole('button', { name: /cubes/ });
		await fireEvent.click(screen.getByRole('button', { name: 'Up one folder' }));
		expect(await screen.findByRole('alert')).toHaveTextContent('outside the folders');
	});
});
