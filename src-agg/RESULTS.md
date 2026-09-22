# Aggregate NYISO/BPA random-forest results

The outage tables and candidate WHEN/WHERE datasets were concatenated with an explicit `operator` column. Outage labels were normalized to `Planned`/`Auto`, and one aggregate truncated-exponential/uniform mixture supplied the foreground/background labels. Geographic zone IDs were never pooled for modelling.

## Shared aggregate decisions

- Aggregated outage rows: 121,688
- Mixture horizon: 7.5 h
- Mixture KS distance: 0.102528
- Mixture foreground probability: 0.642617
- Aggregate model samples: 4,631 foreground / 2,726 background
- Selected event lookback: **1 h**
- Selection rule: lowest mean five-fold aggregate validation MAE; held-out `fold_id == -1` rows were not used.
- Selected-window validation MAE: 418.651 s (SD 8.286 s)

The 7.5 h mixture horizon is a distributional cutoff used to label background events. It is distinct from the 1 h feature lookback selected by model validation.

Normalized event-level outage counts:

| Operator | Planned | Auto |
|---|---:|---:|
| NYISO | 61,739 | 8,340 |
| BPA | 27,926 | 23,683 |

## WHEN: separate held-out comparison

The random baseline draws time-to-event values from the same operator's foreground training distribution. Lower MAE/RMSE is better.

| Operator | Predictor | MAE (min) | RMSE (min) | R² |
|---|---|---:|---:|---:|
| NYISO | Random empirical baseline | 37.525 ± 0.985 | 48.557 ± 1.062 | -1.035 ± 0.089 |
| NYISO | Random forest (ours) | 6.570 | 9.416 | 0.924 |
| BPA | Random empirical baseline | 39.031 ± 1.994 | 48.675 ± 2.085 | -1.127 ± 0.182 |
| BPA | Random forest (ours) | 7.936 | 10.383 | 0.903 |

## WHERE: separate held-out graph-distance comparison

The random baseline draws complete source-local zone pairs from the training distribution. Distances use each operator's previously saved `output/edge_zones.csv` graph topology; lower is better.

| Operator | Predictor | Mean pair distance | Median pair distance | Within one hop |
|---|---|---:|---:|---:|
| NYISO | Random pair baseline | 2.255 ± 0.037 | 2.000 ± 0.000 | 20.0 ± 1.5% |
| NYISO | Random forest (ours) | 1.986 | 2.000 | 29.6% |
| BPA | Random pair baseline | 1.709 ± 0.044 | 1.994 ± 0.053 | 31.6 ± 2.9% |
| BPA | Random forest (ours) | 1.860 | 2.000 | 18.5% |

All reported scores use only each operator's original held-out fold and only events labelled foreground by the shared aggregate mixture. Random forests use 100 trees and `min_samples_leaf=5`. Random baselines report mean ± SD over 200 seeded draws.

Detailed metrics, predictions, feature importances, and serialized models are under `model_outputs/`.
