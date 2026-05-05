# PSC Bridges-2 SLURM scripts

Site-specific launchers used during the original sweeps on PSC Bridges-2
(`cis260099p` allocation). Paths like `/ocean/projects/cis260099p/...`
are hardcoded.

To reuse on another cluster:
1. Replace `PROJECT=/ocean/projects/cis260099p` and the SLURM `-o` log
   path with your local equivalents.
2. Set `INR_ROOT`, `INR_DATA_ROOT`, and the per-dataset cache env vars
   exposed by `paths.txt` (see repo top-level README §1) before invoking.

These scripts are kept for provenance/replication of the original runs;
the portable entrypoints in `scripts/run_*_suite.sh` work on any
multi-GPU node.
