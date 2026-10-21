import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';

import LogView from './LogView.svelte';

/** A fake log endpoint over `logs[stream]`, which a test can append to. */
function fakeLog(logs: Record<string, string>) {
	const calls: string[] = [];
	vi.stubGlobal(
		'fetch',
		vi.fn(async (input: Request) => {
			const url = new URL(input.url);
			const stream = url.searchParams.get('stream') ?? 'stdout';
			const offset = Number(url.searchParams.get('offset') ?? 0);
			calls.push(`${stream}@${offset}`);
			const bytes = new TextEncoder().encode(logs[stream] ?? '');
			const text = new TextDecoder().decode(bytes.slice(offset));
			return new Response(
				JSON.stringify({ text, offset, next_offset: bytes.length, size: bytes.length }),
				{ headers: { 'Content-Type': 'application/json' } }
			);
		})
	);
	return calls;
}

describe('LogView', () => {
	afterEach(() => {
		vi.unstubAllGlobals();
		vi.useRealTimers();
	});

	it("shows a finished run's log and stops reading", async () => {
		const calls = fakeLog({ stdout: 'first line\nsecond line\n' });
		render(LogView, { runId: 'R', runDir: '/runs/R', finished: true, pollMs: 50 });
		expect(await screen.findByText('second line')).toBeInTheDocument();
		expect(screen.getByText('2 lines', { exact: false })).toBeInTheDocument();
		// One last read after the end, then nothing more.
		await waitFor(() => expect(calls.length).toBe(2), { timeout: 2000 });
		await new Promise((r) => setTimeout(r, 300));
		expect(calls.length).toBe(2);
		expect(screen.queryByRole('button', { name: 'Pause' })).toBeNull();
	});

	it('follows a running log, including a line still being written', async () => {
		const logs = { stdout: 'step 1\nstep 2 (half' };
		fakeLog(logs);
		render(LogView, { runId: 'R', runDir: '/runs/R', finished: false, pollMs: 30 });
		expect(await screen.findByText('step 2 (half')).toBeInTheDocument();
		logs.stdout += ' done)\nstep 3\n';
		expect(await screen.findByText('step 3')).toBeInTheDocument();
		expect(screen.getByText('step 2 (half done)')).toBeInTheDocument();
		expect(screen.getByRole('button', { name: 'Pause' })).toHaveAttribute('aria-pressed', 'true');
	});

	it('switches between stdout and stderr', async () => {
		const calls = fakeLog({ stdout: 'out\n', stderr: 'err\n' });
		render(LogView, { runId: 'R', runDir: '/runs/R', finished: true, pollMs: 50 });
		await screen.findByText('out');
		await fireEvent.click(screen.getByRole('button', { name: 'stderr.log' }));
		expect(await screen.findByText('err')).toBeInTheDocument();
		expect(calls).toContain('stderr@0');
		expect(screen.getByRole('link', { name: 'Download' })).toHaveAttribute(
			'href',
			`${location.origin}/api/v2/files/download?path=%2Fruns%2FR%2Fstderr.log`
		);
	});

	it('draws only the lines in view of a long log', async () => {
		const many = Array.from({ length: 5000 }, (_, i) => `line ${i}`).join('\n') + '\n';
		fakeLog({ stdout: many });
		render(LogView, { runId: 'R', runDir: '/runs/R', finished: true, pollMs: 50 });
		await screen.findByText(/5\D?000 lines/);
		const drawn = screen.getByRole('log').querySelectorAll('.whitespace-pre').length;
		expect(drawn).toBeGreaterThan(0);
		expect(drawn).toBeLessThan(200);
	});
});
