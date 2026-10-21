/**
 * What the viewer needs from a FITS header: how many channels, their
 * frequency or velocity, the unit and the beam.
 */

export interface Card {
	key: string;
	value?: unknown;
	comment?: string;
}

export interface CubeInfo {
	shape: number[];
	channels: number;
	unit: string;
	/** "1.56″ × 1.32″", from BMAJ and BMIN (degrees); empty without them. */
	beam: string;
	/** A channel's frequency (GHz) or velocity (km/s), from the spectral axis. */
	channelLabel: (channel: number) => string;
}

/** The header's cards as a lookup. */
function lookup(cards: Card[]): (key: string) => unknown {
	const map = new Map(cards.map((c) => [c.key.toUpperCase(), c.value]));
	return (key) => map.get(key);
}

export function cubeInfo(shape: number[], cards: Card[]): CubeInfo {
	const get = lookup(cards);
	const num = (key: string) => {
		const v = Number(get(key));
		return Number.isFinite(v) ? v : null;
	};
	// numpy order: (..., FREQ, Y, X); the FITS spectral axis is the third.
	const channels = shape.length >= 3 ? shape[shape.length - 3] : 1;
	const ctype = String(get('CTYPE3') ?? '').toUpperCase();
	const crval = num('CRVAL3');
	const cdelt = num('CDELT3');
	const crpix = num('CRPIX3') ?? 1;
	const channelLabel = (channel: number) => {
		if (crval === null || cdelt === null) return '';
		const value = crval + (channel + 1 - crpix) * cdelt;
		if (ctype.startsWith('FREQ')) return `${(value / 1e9).toFixed(4)} GHz`;
		if (ctype.startsWith('VRAD') || ctype.startsWith('VELO') || ctype.startsWith('VOPT')) {
			return `${(value / 1e3).toFixed(1)} km/s`;
		}
		return value.toPrecision(6);
	};
	const bmaj = num('BMAJ');
	const bmin = num('BMIN');
	const beam = bmaj && bmin ? `${(bmaj * 3600).toFixed(2)}″ × ${(bmin * 3600).toFixed(2)}″` : '';
	return { shape, channels, unit: String(get('BUNIT') ?? ''), beam, channelLabel };
}
