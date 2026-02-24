from multiprocessing import Pool
from pathlib import Path

import requests
from tqdm import tqdm

base_path = Path("raw-zip")
base_path.mkdir(exist_ok=True)


def main(urls):
    with Pool(4) as pool:
        for resp in tqdm(pool.imap_unordered(requests.get, urls), total=len(urls)):
            if resp is None:
                continue
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}: {resp.url}")
                continue

            fname = resp.url.split("/")[-1] or "download.bin"
            path = base_path / fname
            path.write_bytes(resp.content)


if __name__ == "__main__":
    urls = [
        f"https://mis.nyiso.com/public/csv/realtimelineoutages/{year}{month:02d}01RTLineOutages_csv.zip"
        for year in range(2008, 2027)
        for month in range(1, 13)
    ]
    main(urls)
