# Tune Report Data Archive

- Created: 2026-03-03T13:42:41
- Source sweep: `/home/jhoya/Workspaces/ai-22-cv-cv-team1/members/hyjeon1985/outputs/hpc_proxy/date=2026-03-03/sweep_id=09-13-17`
- W&B group filter: `tune__date=2026-03-03__sweep_id=09-13-17`

## Included
- `sweep/`: multirun and tune aggregate files (`tune_results.csv`, `tune_summary.json`)
- `trials/`: per-run lightweight files (`train.json`, `eval.json`, `prep.json`, `.hydra/*`)
- `wandb_runs/`: local W&B files for matching tune runs (`config.yaml`, `wandb-summary.json`)
- `derived/`: flattened tables for report charts

## Excluded
- checkpoints and other heavy artifacts
- copied dataset/images
- hardware/system metadata (`wandb-metadata.json`)

## Quick Start
- trial table: `derived/trial_metrics_flat.csv`
- wandb table: `derived/wandb_summary_flat.csv`
- selector summary: `sweep/tune_summary.json`
