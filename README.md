# bpa-nyiso-data
Clean nyiso and bpa datasets.

Ordered scrips/notebooks to clean dataset:

0- [`download.py`](download.py): Download raw zip files from `nyios.com`.

1- [`processing_actual_logs.ipynb`](processing_actual_logs.ipynb): Creates a csv containing processed outages.

2- [`processing_scheduled_logs.ipynb`](processing_scheduled_logs.ipynb): Compress scheduled outage plans.

3- [`compare_processed_outages.ipynb`](compare_processed_outages.ipynb): Compares extracted records by our algorithm with that of extracted by Carrington/Dobson. Also removes unconnected components of the graph.

4- [`set_outage_type.ipynb`](set_outage_type.ipynb): Adding "OutageType" to actual outages by comparing its records on the scheduled outages.



## other scripts and files:

- [`data.ipynb`](data.ipynb): EDA on Carrington/Dobson data.
- [`our_data_eda.ipynb`](our_data_eda.ipynb): EDA on our cleaned data.
