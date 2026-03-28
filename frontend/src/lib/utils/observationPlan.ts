export interface ObservationConfigInput {
	name: string;
	array_type: '12m' | '7m' | 'TP';
	antenna_array: string;
	total_time_s: number;
	correlator?: string;
}

export function deriveArrayType(antennaArray: string): string {
	const upper = (antennaArray || '').toUpperCase();
	const has12m = upper.includes('DA') || upper.includes('DV');
	const has7m = upper.includes('CM');
	const hasTp = upper.includes('PM');

	if (has12m && !has7m && !hasTp) return '12m';
	if (has7m && !has12m && !hasTp) return '7m';
	if (hasTp && !has12m && !has7m) return 'TP';
	if (has12m && has7m && hasTp) return '12m+7m+TP';
	if (has12m && has7m) return '12m+7m';
	if (has12m && hasTp) return '12m+TP';
	if (has7m && hasTp) return '7m+TP';
	return '12m';
}

export function splitAntennaArrayByType(antennaArray: string): Array<[ObservationConfigInput['array_type'], string]> {
	const tokens = (antennaArray || '')
		.split(/\s+/)
		.map((token) => token.trim())
		.filter(Boolean);

	const groups: Record<ObservationConfigInput['array_type'], string[]> = {
		'12m': [],
		'7m': [],
		TP: []
	};

	for (const token of tokens) {
		const upper = token.toUpperCase();
		if (upper.includes('CM')) {
			groups['7m'].push(token);
		} else if (upper.includes('PM')) {
			groups.TP.push(token);
		} else {
			groups['12m'].push(token);
		}
	}

	const ordered: Array<[ObservationConfigInput['array_type'], string]> = [];
	for (const arrayType of ['12m', '7m'] as const) {
		if (groups[arrayType].length > 0) {
			ordered.push([arrayType, groups[arrayType].join(' ')]);
		}
	}
	return ordered;
}

export function inferObservationConfigsFromMetadataRow(
	row: Record<string, unknown>,
	totalTimeS: number,
	correlator?: string
): ObservationConfigInput[] | undefined {
	const rawAntennaArray = String(row.antenna_arrays ?? row.antenna_array ?? '').trim();
	if (!rawAntennaArray) {
		return undefined;
	}

	const splitGroups = splitAntennaArrayByType(rawAntennaArray);
	if (splitGroups.length === 0) {
		return undefined;
	}

	return splitGroups.map(([arrayType, antennaGroup], index) => ({
		name: `config_${index}_${arrayType}`,
		array_type: arrayType,
		antenna_array: antennaGroup,
		total_time_s: totalTimeS,
		correlator
	}));
}
