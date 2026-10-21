import { describe, expect, it } from 'vitest';

import { cubeInfo } from './fitsHeader';

const card = (key: string, value: unknown) => ({ key, value, comment: '' });

describe('cubeInfo', () => {
	it('reads the channels, their frequencies, the unit and the beam', () => {
		const info = cubeInfo(
			[1, 240, 512, 512],
			[
				card('BUNIT', 'Jy/beam'),
				card('CTYPE3', 'FREQ'),
				card('CRVAL3', 107.6e9),
				card('CDELT3', 7.8125e6),
				card('CRPIX3', 1),
				card('BMAJ', 2.29 / 3600),
				card('BMIN', 1.8 / 3600)
			]
		);
		expect(info.channels).toBe(240);
		expect(info.unit).toBe('Jy/beam');
		expect(info.beam).toBe('2.29″ × 1.80″');
		expect(info.channelLabel(0)).toBe('107.6000 GHz');
		expect(info.channelLabel(128)).toBe('108.6000 GHz');
	});

	it('handles images, velocity axes and missing keywords', () => {
		const image = cubeInfo([512, 512], []);
		expect(image.channels).toBe(1);
		expect(image.beam).toBe('');
		expect(image.channelLabel(0)).toBe('');
		const velocity = cubeInfo(
			[10, 64, 64],
			[card('CTYPE3', 'VRAD'), card('CRVAL3', -5000), card('CDELT3', 1000), card('CRPIX3', 1)]
		);
		expect(velocity.channelLabel(5)).toBe('0.0 km/s');
	});
});
