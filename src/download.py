from multiprocessing import Pool
from pathlib import Path

import requests
from tqdm import tqdm


def main(urls, base_path):
    with Pool(4) as pool:
        for resp in tqdm(pool.imap_unordered(requests.get, urls), total=len(urls)):
            if resp is None:
                continue
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}: {resp.url}")
                continue

            fname = resp.url.split("/")[-1]
            path = base_path / fname
            path.write_bytes(resp.content)


if __name__ == "__main__":
    actual_outages_urls = [
        f"https://mis.nyiso.com/public/csv/realtimelineoutages/{year}{month:02d}01RTLineOutages_csv.zip"
        for year in range(2008, 2027)
        for month in range(1, 13)
    ]
    actual_outages_base_path = Path("raw-zip")
    actual_outages_base_path.mkdir(exist_ok=True)
    main(actual_outages_urls, actual_outages_base_path)

    scheduled_outages_urls = [
        f"http://mis.nyiso.com/public/csv/schedlineoutages/{year}{month:02d}01SCLineOutages_csv.zip"
        for year in range(2008, 2027)
        for month in range(1, 13)
    ]
    scheduled_outages_base_path = Path("raw-zip-scheduled")
    scheduled_outages_base_path.mkdir(exist_ok=True)
    main(scheduled_outages_urls, scheduled_outages_base_path)
