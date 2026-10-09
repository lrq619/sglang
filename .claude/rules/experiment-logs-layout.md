# Experiment logs: one flat directory per run under `logs/`

`logs/` (symlink to `~/scratch/logs`) holds one directory per run, all as
direct children. The user's visualization tool only lists `logs/*/`, so a run
nested one level deeper is invisible to it.

- **Never group runs into a parent folder** — not per sweep, not per date, not
  per dataset. A sweep of N runs produces N sibling directories in `logs/`.
- Encode the grouping in the directory **name** instead, with a shared prefix:
  `logs/<sweep_tag>_r<rate>_<timestamp>/` (e.g. `sweep_20261008_135154_r12_20261008_150524`).
- Sweep-level bookkeeping (the sweep's progress log) goes in a flat file next to
  the runs, `logs/<sweep_tag>.log`, never in a directory.
- Each run directory is self-contained: `loadgen.csv` (request metrics),
  `metrics.csv` (server time series), `servers/`, `run_info.txt`, `dry_run.log`.

`loadgen/sweep.sh` and `loadgen/dry_run.sh` already follow this layout via
`LOG_ROOT` / `SWEEP_TAG` / `RUN_NAME`; keep any new experiment script consistent.
