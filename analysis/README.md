# Revision analysis (reproducible)

Python scripts that regenerate every derived number, table, and figure added in
the revision of *Beyond Leaderboards: Deployment-Context-Aware Evaluation of
Small and Large Language Models* (IEEE Access, Access-2026-43998).

## Requirements

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Run

```
python recover_figure_values.py     # optional: re-derive data/recovered_from_figures.csv
python build_dataset.py             # -> data/aggregated_metrics.csv (+ provenance per field)
python make_revision_outputs.py     # -> ../figures/rev_*.pdf, ../tables/rev_*.tex, data/derived_*.csv
python compute_supplementary.py --repo /path/to/TB-2026-SLM-benchmarking
                                    # -> ../supplementary/s*.tex, data/derived_run_level_intervals.csv,
                                    #    data/derived_energy_intervals.csv, data/derived_mcnemar.csv
python -m pytest tests -q           # unit tests (reference values for Wilson/Newcombe/Holm)
```

## Inputs and provenance

`data/aggregated_metrics.csv` records the source of every value in a
`<field>_source` column, using this precedence:

1. `tables`: LaTeX tables exported by the results pipeline (`../tables/table_0*.tex`).
2. `figure`: values recovered exactly from the vector PDF figures produced by the
   same pipeline (`recover_figure_values.py`). Agreement with overlapping table
   values is checked on every build (max. deviation 0.09%).
3. `thesis`: Tables 8.1 and 8.2 of the B.Sc. thesis (cached in
   `data/thesis_tables_8_1_8_2.csv`).

Once the raw per-sample logs are exported, prefer them over this aggregated file.

## Supplementary Tables S1-S5

`compute_supplementary.py` reads the per-run result files of the benchmarking
repository (`results/{pi5,cpu,vllm}/<model>/<session>/*_samples.csv` and the session
summaries), reproduces every published mean, and computes the run-level intervals,
energy intervals, accuracy spread, and paired McNemar tests of the supplementary
material. Run it before `make_revision_outputs.py` (Table 6 uses its request rates).

## Generic raw-log analysis

`run_level_uncertainty.py` works on the framework's per-sample CSV files. It
computes item-level bootstrap and Wilson intervals, paired exact McNemar tests
with Holm adjustment, run-level t-intervals and two-stage cluster-bootstrap
intervals (mean and p95) for TTFT, latency, tokens/s, and energy, as well as the
run manifest and the pairwise item overlap:

```
python run_level_uncertainty.py --samples "<results_dir>/**/*samples*.csv" --out derived_raw \
    --col-item <item id column> --col-correct <correctness column> ...
```

Use `--help` to see the column-name options.
