/**
 * The simulation workflow: one `almasim simulation` run per chosen metadata
 * row, and what the estimate says the batch will take.
 */
import { buildArgv, initialValues, type Command, type Values } from './commandForm';
import { RUN_DIR } from './runDir';

/** The simulation form's fields the page sets from the chosen rows. */
export const ROW_FIELDS = ['metadata_path', 'row_idx'];

/**
 * The console's starting values over the CLI's. The CLI writes an ML shard
 * to one file in your home and keeps nothing else; the console runs one job
 * per row at once, so each writes its shard, and keeps its FITS products,
 * in its own run folder.
 */
export const CONSOLE_DEFAULTS: Values = {
	output_dir: `${RUN_DIR}/outputs`,
	ml_shard_path: `${RUN_DIR}/ml_shard.h5`,
	save_mode: 'fits',
	persist_standard_outputs: true
};

/** The form's starting values: the CLI's, then the console's where the command has the field. */
export function simulationDefaults(command: Command): Values {
	const values = initialValues(command);
	for (const [name, value] of Object.entries(CONSOLE_DEFAULTS)) {
		if (name in values) values[name] = value;
	}
	return values;
}

/**
 * One run's command line: the form, for one row of the metadata CSV. In a
 * batch, a shared output folder gets a folder per row, so runs that go at
 * once never write over each other.
 */
export function simulationArgv(
	command: Command,
	values: Values,
	metadataPath: string,
	rowIdx: number,
	batch = false
): string[] {
	const run: Values = { ...values, metadata_path: metadataPath, row_idx: String(rowIdx) };
	const out = typeof values.output_dir === 'string' ? values.output_dir : '';
	if (batch && out && !out.includes(RUN_DIR))
		run.output_dir = `${out.replace(/\/+$/, '')}/row-${rowIdx}`;
	return buildArgv(command, run);
}

/** The estimate for the batch: per simulation, times runs per row, times rows. */
export function batchGb(perSimulationGb: number, rows: number, perRow: number): number {
	return perSimulationGb * Math.max(rows, 0) * Math.max(perRow, 1);
}
