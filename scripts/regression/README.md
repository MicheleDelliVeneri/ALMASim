# Regression gate (MADIV-28)

Every change to the pipeline must leave its products unchanged, unless the PR
says otherwise. These scripts check that. They compare every product **bit for
bit** with a versioned baseline.

## The three tiers

| Tier | Where | What runs | Compared with |
|---|---|---|---|
| 1 | GitHub default runner, every PR that touches pipeline code (`golden-ci.yml`, jobs `tier1` and `simulation`) | The trimmed golden window (release `test-data-v1`) through extract → calibrate → compute-parameters → image → clean-quality (`ci_chain.sh`), plus four seeded simulations (`sim_golden.sh`) | Release `golden-baseline-vN`, as named in `ci/golden/baseline.json` |
| 2 | GitHub default runner, nightly and on Slurm-related PRs (job `tier2`) | The same chain through a single-node Slurm cluster in the container (`ci_slurm_cluster.sh`, `BACKEND=slurm`) | The tier-1 baseline |
| 3 | zwicky, submitted with `sbatch` for every pipeline PR (not CI) | The full 13 GB golden window `uid___A002_X113d9b7_Xb670` SPW-31_FIELD-2, through importasdm → calibrate → compute-parameters → image (`golden_run.sbatch`; the imaging parameters are computed, not taken from MADIV-16) | `BASELINE_DIR` in `golden.env` (v2-6cb0829e; v0 = ms-raw, ms-calibrated-flagged and MADIV-16 is kept as the scientific reference) |

Imaging is single-threaded in all tiers. WSClean with `-j N` is not
bit-reproducible: two `-j 4` runs differ by up to 4e-3 × std. Unpack,
calibration and `-j 1` imaging are reproducible across runs, nodes and GitHub
runners.

## Running the full-size gate (tier 3)

```bash
scripts/regression/submit_golden.sh HEAD "PR #123"
```

This exports the commit into `/SKAO_data/cycle11/MADIV-28/regression/<date>-<sha>/`,
submits `golden_run.sbatch` and leaves `report.md` in `regression/logs/`. A
passing run deletes its directory. A failing run keeps its report, logs,
digests and image products in `regression/failed/`. Useful environment
variables (pass them through `SBATCH_EXTRA="--export=ALL,..."`):

- `GOLDEN_KEEP=1`: keep a passing run's products as a candidate baseline.
- `GOLDEN_FROM=compare`: rerun only the comparison on an existing run directory.
- `GOLDEN_BASELINE=DIR`: compare with another baseline directory.
- `GOLDEN_CORES=N`: imaging threads (default 1).

## When products are meant to change

1. Label the PR `golden:supersede` and describe the change under
   **Expected product changes** in the PR description. The `golden-labels`
   check enforces both.
2. CI then reports the differences instead of failing, and uploads the
   candidate baselines as artifacts. Reviewers check that the differences match
   the description.
3. Adopt the CI baseline from the PR's latest golden-ci run, and commit the
   updated `ci/golden/baseline.json` to the PR:

   ```bash
   scripts/regression/adopt_candidate_baseline.sh RUN_ID v2 "SIM-3: integration time units (#123)"
   ```

4. For the full-size baseline, run the gate with `GOLDEN_KEEP=1`, then
   `promote_baseline.sh regression/kept/<run> v2-<sha> "reason"`, and point
   `BASELINE_DIR` in `golden.env` at it in the same PR.

`golden:skip` skips the CI jobs. It needs a reason in the PR body and
maintainer approval, and the next nightly run on `main` must pass.

## Files

| File | Purpose |
|---|---|
| `compare_products.py` | Compares tables (column SHA-256), FITS files (data, headers, beam tables), JSON, CSV and task logs |
| `ci_chain.sh` | The pipeline chain on the CI fixture (tiers 1 and 2) |
| `sim_golden.sh` | Seeded simulations for the simulation golden |
| `ci_slurm_cluster.sh` | Single-node Slurm cluster inside the CI container (tier 2) |
| `build_ci_fixture.py` | Builds the trimmed fixture from the full golden window |
| `package_baseline.sh` | Packages a run's products as a baseline bundle |
| `adopt_candidate_baseline.sh` | Publishes a supersede PR's candidate as `golden-baseline-vN` |
| `golden.env`, `golden_run.sbatch`, `submit_golden.sh`, `promote_baseline.sh` | The full-size gate on zwicky |
