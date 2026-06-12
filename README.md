# bpa-nyiso-data
Clean nyiso and bpa datasets.

Ordered scrips/notebooks to clean dataset:

0- [`download.py`](download.py): Download raw zip files from `nyios.com`.

1- [`processing_actual_logs.ipynb`](processing_actual_logs.ipynb): Creates a csv containing processed outages.

2- [`processing_scheduled_logs.ipynb`](processing_scheduled_logs.ipynb): Compress scheduled outage plans.

3- [`compare_processed_outages.ipynb`](compare_processed_outages.ipynb): Compares extracted records by our algorithm with that of extracted by Carrington/Dobson. Also removes unconnected components of the graph.

4- [`set_outage_type.ipynb`](set_outage_type.ipynb): Adding "OutageType" to actual outages by comparing its records on the scheduled outages.

5- [`create_dataset.ipynb`](create_dataset.ipynb): Create dataset using preprocessed csv files. You should change the `EVENT_WINDOW_HOURS` and rerun the script for different window sizes.

6- [`auto_outage_distribution.ipynb`](auto_outage_distribution.ipynb): Check auto outage distribution in time and reject null hypothesis (if auto outages are independent of planned outages, their distribution over time is uniform).

7- [`predictive_maintenance.ipynb`](src/predictive_maintenance.ipynb): Train and compare decision-tree and random-forest models across multiple event-window sizes to predict the time and endpoint zones of the next automatic outage. Evaluate asymmetric early/late prediction costs, time-interval classification, and zone-distance errors, and analyze zone-model feature importance with SHAP.

## other scripts and files:

- [`data.ipynb`](data.ipynb): EDA on Carrington/Dobson data.
- [`our_data_eda.ipynb`](our_data_eda.ipynb): EDA on our cleaned data.
