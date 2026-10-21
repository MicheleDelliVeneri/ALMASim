import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';

import ImageViewer from './ImageViewer.svelte';
import QualityView from './QualityView.svelte';

/** A fake console serving one 3-channel cube. */
function fakeCube() {
	const planes: number[] = [];
	vi.stubGlobal(
		'fetch',
		vi.fn(async (input: Request) => {
			const url = new URL(input.url);
			const json = (body: unknown) =>
				new Response(JSON.stringify(body), { headers: { 'Content-Type': 'application/json' } });
			if (url.pathname === '/api/v2/files/fits') {
				return json({
					path: '/data/cube.fits',
					hdus: 1,
					hdu: 0,
					shape: [1, 3, 4, 4],
					cards: [
						{ key: 'BUNIT', value: 'Jy/beam', comment: '' },
						{ key: 'CTYPE3', value: 'FREQ', comment: '' },
						{ key: 'CRVAL3', value: 100e9, comment: '' },
						{ key: 'CDELT3', value: 1e9, comment: '' },
						{ key: 'CRPIX3', value: 1, comment: '' }
					]
				});
			}
			const channel = Number(url.searchParams.get('channel'));
			planes.push(channel);
			return json({
				path: '/data/cube.fits',
				shape: [4, 4],
				step: 1,
				channel,
				min: 0,
				max: 15,
				data: [
					[0, 1, 2, 3],
					[4, 5, 6, 7],
					[8, 9, 10, 11],
					[12, 13, 14, null]
				]
			});
		})
	);
	return planes;
}

describe('ImageViewer', () => {
	afterEach(() => vi.unstubAllGlobals());

	it('reads the cube, shows its channels, and steps through them from the keyboard', async () => {
		const planes = fakeCube();
		render(ImageViewer, { path: '/data/cube.fits' });
		expect(await screen.findByText('Channel 1 / 3')).toBeInTheDocument();
		expect(screen.getByText('100.0000 GHz')).toBeInTheDocument();
		expect(screen.getByText('Jy/beam')).toBeInTheDocument();
		await waitFor(() => expect(planes).toEqual([0]));

		const image = screen.getByLabelText(/cube\.fits, channel 1/);
		await fireEvent.keyDown(image, { key: ']' });
		expect(await screen.findByText('Channel 2 / 3')).toBeInTheDocument();
		expect(screen.getByText('101.0000 GHz')).toBeInTheDocument();
		await waitFor(() => expect(planes).toEqual([0, 1]));
		await fireEvent.keyDown(image, { key: '[' });
		await fireEvent.keyDown(image, { key: '[' }); // stops at the first
		expect(await screen.findByText('Channel 1 / 3')).toBeInTheDocument();
	});
});

describe('QualityView', () => {
	it('lists each section, rounding the figures', () => {
		render(QualityView, {
			quality: {
				task: '/out/SPW-25',
				continuum: {
					snr: 42.353924511367275,
					beam_arcsec: [3.479, 2.463],
					model_components: 43,
					residual_rings_sigma: { '0-0.5': 0.9315697 }
				}
			}
		});
		expect(screen.getByRole('heading', { name: 'continuum' })).toBeInTheDocument();
		expect(screen.getByText('S/N')).toBeInTheDocument();
		expect(screen.getByText('42.4')).toBeInTheDocument();
		expect(screen.getByText('3.48 × 2.46')).toBeInTheDocument();
		expect(screen.getByText('43')).toBeInTheDocument();
		expect(screen.getByText('0-0.5')).toBeInTheDocument();
		expect(screen.queryByText('/out/SPW-25')).toBeNull();
	});
});
