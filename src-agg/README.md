# Aggregated NYISO/BPA workflow

This folder implements the shared-data/separate-model experiment:

1. Normalize and concatenate the NYISO and BPA event tables, retaining an
   explicit `operator` field and source-prefixed line/bus IDs.
2. Concatenate every aligned WHEN/WHERE candidate dataset.
3. Fit one foreground/background outage-regime model on the combined data.
4. Select one shared `event_window_hours` using five-fold random-forest
   validation on the combined foreground rows. The held-out `fold_id == -1`
   rows are excluded from selection.
5. Split on `operator`, train independent NYISO and BPA random forests for
   WHEN and WHERE, and report their held-out results separately.

The mixture horizon and `event_window_hours` are intentionally distinct. The
former labels likely background outages; the latter controls how much event
history is represented in the model features.

Run from the repository root:

```bash
python src-agg/pipeline.py
```

Generated data are written to `src-agg/output/`. Selection metadata, trained
models, predictions, and detailed metrics are written to
`src-agg/model_outputs/`; the concise result table is in `src-agg/RESULTS.md`.

Generate the paper-style figures with:

```bash
python src-agg/generate_pdfs.py
```

All figures are written to `src-agg/pdfs/`; its README maps each PDF to the
corresponding figure family in the main project README.

Zone IDs and graph distances are operator-local. They are never treated as a
shared NYISO/BPA label space; the aggregate stage is used only for outage-regime
labelling and shared window selection.
