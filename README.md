# bpa-nyiso-data
Clean nyiso and bpa datasets.

Ordered scrips/notebooks to clean dataset:

0- [`download.py`](download.py): Download raw zip files from `nyiso.com`.

1- [`processing_actual_logs.ipynb`](src/processing_actual_logs.ipynb): Creates a csv containing processed outages.

2- [`processing_scheduled_logs.ipynb`](src/processing_scheduled_logs.ipynb): Compress scheduled outage plans.

3- [`compare_processed_outages.ipynb`](src/compare_processed_outages.ipynb): Compares extracted records by our algorithm with that of extracted by Carrington/Dobson. Also removes unconnected components of the graph.

4- [`set_outage_type.ipynb`](src/set_outage_type.ipynb): Adding "OutageType" to actual outages by comparing its records on the scheduled outages.

5- [`create_dataset.ipynb`](src/create_dataset.ipynb): Create dataset using preprocessed csv files. You should change the `EVENT_WINDOW_HOURS` and rerun the script for different window sizes.

6- [`clean_dataset.ipynb`](src/clean_dataset.ipynb): Fit a truncated-exponential and uniform mixture to the time-to-event labels, select the best fitting window using the Kolmogorov-Smirnov distance, and mark background samples in all WHEN and WHERE datasets.

7- [`auto_outage_distribution.ipynb`](src/auto_outage_distribution.ipynb): Check auto outage distribution in time and reject null hypothesis (if auto outages are independent of planned outages, their distribution over time is uniform).

8- [`predictive_maintenance.ipynb`](src/predictive_maintenance.ipynb): Train and compare decision-tree and random-forest models across multiple event-window sizes to predict the time and endpoint zones of the next automatic outage. Evaluate asymmetric early/late prediction costs, time-interval classification, and zone-distance errors, and analyze zone-model feature importance with SHAP.

9- [`predictive_maintenance_xgboost_when.ipynb`](src/predictive_maintenance_xgboost_when.ipynb): Select an event-window size and tune an XGBoost regressor to predict the time until the next automatic outage. Compare standard and late-penalized models using timing-error metrics and distributions.

10- [`predictive_maintenance_transformer_where.ipynb`](src/predictive_maintenance_transformer_where.ipynb): Train Transformer models to predict the unordered pair of endpoint zones for the next automatic outage. Compare cross-entropy and graph-distance-aware losses, with average and self-attentive pooling, using exact-pair accuracy and graph-distance error.

11- [`interpretability.ipynb`](src/interpretability.ipynb): Interpret the trained WHERE Transformer's attention across time blocks and the WHEN XGBoost model using highest-gain tree visualizations and global and local SHAP analyses.

## other scripts and files:

- [`data.ipynb`](src/data.ipynb): EDA on Carrington/Dobson data.
- [`our_data_eda.ipynb`](src/our_data_eda.ipynb): EDA on our cleaned data.
- [`count_events.ipynb`](src/count_events.ipynb): Inspect raw actual-outage archives and summarize log sizes, unique devices, transmission lines, and total event counts.
- [`graph_degree.ipynb`](src/graph_degree.ipynb): Build and export the bus-level outage graph, match bus names to geographic locations, and compare planned and automatic outage frequencies across graph k-cores.
- [`graph_preprocessing.ipynb`](src/graph_preprocessing.ipynb): Analyze the outage graph's node-degree distribution using a histogram, log-log complementary cumulative distribution, and power-law fit.
