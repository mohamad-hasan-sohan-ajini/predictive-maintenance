# bpa-nyiso-data
Clean nyiso and bpa datasets.

Ordered scrips/notebooks to clean dataset:

0- [`download.py`](download.py): Download raw zip files from `nyiso.com`.

1- [`processing_actual_logs.ipynb`](src/processing_actual_logs.ipynb): Creates a csv containing processed outages.

2- [`processing_scheduled_logs.ipynb`](src/processing_scheduled_logs.ipynb): Compress scheduled outage plans.

3- [`compare_processed_outages.ipynb`](src/compare_processed_outages.ipynb): Compares extracted records by our algorithm with that of extracted by Carrington/Dobson. Also removes unconnected components of the graph.

4- [`set_outage_type.ipynb`](src/set_outage_type.ipynb): Adding "OutageType" to actual outages by comparing its records on the scheduled outages.

5- [`create_dataset.ipynb`](src/create_dataset.ipynb): Create dataset using preprocessed csv files. You should change the `EVENT_WINDOW_HOURS` and rerun the script for different window sizes.

6- [`auto_outage_distribution.ipynb`](src/auto_outage_distribution.ipynb): Check auto outage distribution in time and reject null hypothesis (if auto outages are independent of planned outages, their distribution over time is uniform).

7- [`predictive_maintenance.ipynb`](src/predictive_maintenance.ipynb): Train and compare decision-tree and random-forest models across multiple event-window sizes to predict the time and endpoint zones of the next automatic outage. Evaluate asymmetric early/late prediction costs, time-interval classification, and zone-distance errors, and analyze zone-model feature importance with SHAP.

## other scripts and files:

- [`data.ipynb`](src/data.ipynb): EDA on Carrington/Dobson data.
- [`our_data_eda.ipynb`](src/our_data_eda.ipynb): EDA on our cleaned data.
- [`count_events.ipynb`](src/count_events.ipynb): Inspect raw actual-outage archives and summarize log sizes, unique devices, transmission lines, and total event counts.
- [`graph_degree.ipynb`](src/graph_degree.ipynb): Build and export the bus-level outage graph, match bus names to geographic locations, and compare planned and automatic outage frequencies across graph k-cores.
- [`graph_preprocessing.ipynb`](src/graph_preprocessing.ipynb): Analyze the outage graph's node-degree distribution using a histogram, log-log complementary cumulative distribution, and power-law fit.
