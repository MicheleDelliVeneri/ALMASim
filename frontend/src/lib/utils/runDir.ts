/**
 * `{run_dir}` in an argument is the run's own folder, filled in when the job
 * runs (almasim.jobs.model.RUN_DIR): a command can write its results there.
 */
export const RUN_DIR = '{run_dir}';

/** `path` with the placeholder replaced by `runDir`. */
export function inRunDir(path: string, runDir: string): string {
	return path.split(RUN_DIR).join(runDir);
}
