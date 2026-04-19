from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
from multiprocessing.pool import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import pandas as pd
from PIL import Image


TIMEOUT = 30            # seconds per request before giving up
THREADS_PER_WORKER = 4  # concurrent downloads inside each process
MAX_RETRY_ROUNDS = 3    # how many retry passes before abandoning failures
RETRY_BACKOFF = 10      # seconds to wait between retry rounds

logging.basicConfig(
    level=logging.INFO,
    format="[%(processName)s] %(message)s",
)
log = logging.getLogger(__name__)


def make_session():
    """Create a requests Session with connection pooling and automatic retries
    for transient network errors (e.g. 500/502/503/504)."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_single_image(args):
    """
    Download all 4 colour channels for one image ID.
    Returns the image ID on failure, None on success.
    """
    i, base_url, save_dir, image_size, session = args
    img_id = i.split('_', 1)

    for color in ("red", "green", "blue", "yellow"):
        img_path = img_id[0] + '/' + img_id[1] + '_' + color + '.jpg'
        img_name = i + '_' + color + '.png'
        out_filename = os.path.join(save_dir, img_name)

        if os.path.exists(out_filename):
            continue

        img_url = base_url + img_path
        try:
            r = session.get(img_url, allow_redirects=True, stream=True, timeout=TIMEOUT)
            r.raise_for_status()
            r.raw.decode_content = True

            im = Image.open(r.raw)
            im = im.resize(image_size, Image.LANCZOS).convert('L')
            im.save(out_filename, 'PNG')
        except Exception as e:
            log.warning("FAILED %s: %s", img_url, e)
            return i   # signal failure

    return None  # success


def download(pid, image_list, base_url, save_dir, image_size=(512, 512)):
    """Worker function: downloads a shard of the image list.

    Uses a thread pool internally so network I/O is done concurrently,
    then retries failures up to MAX_RETRY_ROUNDS times with a backoff.
    """
    session = make_session()

    for attempt in range(1, MAX_RETRY_ROUNDS + 1):
        if not image_list:
            break

        log.info("Attempt %d/%d — %d images to download",
                 attempt, MAX_RETRY_ROUNDS, len(image_list))

        failed = []
        args = [(i, base_url, save_dir, image_size, session) for i in image_list]

        with ThreadPoolExecutor(max_workers=THREADS_PER_WORKER) as executor:
            futures = {executor.submit(download_single_image, a): a[0] for a in args}
            with tqdm(total=len(image_list), desc=f"Worker {pid}", position=int(pid),
                      leave=True) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        failed.append(result)
                    pbar.update(1)

        log.info("Worker %s: %d failed after attempt %d", pid, len(failed), attempt)
        image_list = failed

        if image_list and attempt < MAX_RETRY_ROUNDS:
            log.info("Worker %s: waiting %ds before retry...", pid, RETRY_BACKOFF)
            time.sleep(RETRY_BACKOFF)

    if image_list:
        log.warning("Worker %s: gave up on %d images: %s", pid, len(image_list), image_list[:5])


def main():
    process_num = 24
    image_size = (512, 512)
    url = 'https://images.proteinatlas.org/'
    csv_path = "data/HPAv18RBGY_wodpl.csv"
    save_dir = "data/raw/external"

    os.makedirs(save_dir, exist_ok=True)

    log.info("Parent process %s", os.getpid())
    img_list = list(pd.read_csv(csv_path)['Id'])
    img_splits = np.array_split(img_list, process_num)
    assert sum(len(v) for v in img_splits) == len(img_list)

    log.info("Downloading %d images across %d workers (%d threads each)",
             len(img_list), process_num, THREADS_PER_WORKER)

    p = Pool(process_num)
    for i, split in enumerate(img_splits):
        p.apply_async(
            download, args=(str(i), list(split), url, save_dir, image_size)
        )

    log.info("Waiting for all subprocesses done...")
    p.close()
    p.join()
    log.info("All subprocesses done.")


if __name__ == '__main__':
    main()