# Repeat automatic-outage prediction

`main.py` asks whether recent, same-line outage history predicts another
automatic outage on that line within the next 30 days. It runs the same
random-forest pipeline on:

- NYISO: `../src/processed-actual-outages.csv`
- BPA: `../src-bpa/bpa_processed.csv`

For every source record, the candidate features count strictly earlier
same-line records in 0.5, 1, 2, 3, or 4 hour lookback windows. Records sharing
the anchor timestamp are excluded from one another's features, preventing
same-time leakage. The target is one when a strictly later automatic outage
for the same line occurs within 30 days.

No records are deduplicated and no gap-based refinement/filter is applied.
Rows lacking a usable timestamp or a full 30-day observation period at the
end of a source remain accounted for in `run_metadata.json`, but cannot supply
a trustworthy supervised label.

The split is chronological (60% train, 20% validation, 20% test, with equal
timestamps kept together). Validation average precision selects the history
window. The classification threshold is selected on validation data for F1;
the selected model is refit on train plus validation and evaluated once on
the test period. The random seed is fixed at `20260922`.

Run both systems from the repository root:

```bash
python src-same/main.py
```

For a quick smoke test with fewer trees:

```bash
python src-same/main.py --n-estimators 10
```

Outputs are written to `src-same/outputs/` by default. Each system gets window
validation metrics, test predictions, a held-out ROC curve in
`test_roc_curve.pdf`, random-forest feature importances, the fitted model, and
run metadata. `test_summary.csv` compares final NYISO and BPA test performance.
