"""
Phase 1: Dataset Downloader Module
==================================
Handles parallel, robust downloading of the 53 patient recordings from the
PhysioNet BIDMC PPG and Respiration Dataset.
"""

import time
import urllib.request
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict

from config.settings import DATA_DIR, PHYSIONET_BASE_URL, PATIENT_IDS

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BDA-Research/1.0"}

def _fetch_single_file(item: Tuple[str, Path], timeout: int = 20, max_retries: int = 3) -> Tuple[bool, str, int]:
    """Downloads a single file with exponential backoff retry logic."""
    url, dest = item
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read()
                temp_path = dest.with_suffix(dest.suffix + ".downloading")
                with open(temp_path, "wb") as f:
                    f.write(content)
                temp_path.replace(dest)
            return True, dest.name, len(content)
        except Exception as e:
            if attempt == max_retries - 1:
                return False, dest.name, 0
            time.sleep(1.0 * (attempt + 1))
    return False, dest.name, 0

def download_bidmc_dataset(max_workers: int = 8) -> Dict[str, int]:
    """
    Downloads all 53 ICU patient records from PhysioNet in parallel.
    Skips files that are already present and non-empty.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    extensions = [".hea", ".dat", ".breath", "n.hea", "n.dat"]
    download_queue: List[Tuple[str, Path]] = []

    for patient in PATIENT_IDS:
        for ext in extensions:
            filename = f"{patient}{ext}"
            file_path = DATA_DIR / filename
            if not file_path.exists() or file_path.stat().st_size == 0:
                download_queue.append((f"{PHYSIONET_BASE_URL}{filename}", file_path))

    if not download_queue:
        print(f"[Phase 1 Downloader] All 53 records already present in {DATA_DIR}. Skipping download.")
        return {"total": 53 * len(extensions), "downloaded": 0, "failed": 0}

    print(f"[Phase 1 Downloader] Starting parallel download of {len(download_queue)} missing files ({max_workers} threads)...")
    start_time = time.time()
    succeeded = 0
    failed = 0
    total_bytes = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_single_file, task): task for task in download_queue}
        for future in concurrent.futures.as_completed(futures):
            ok, fname, byte_count = future.result()
            if ok:
                succeeded += 1
                total_bytes += byte_count
                if succeeded % 25 == 0 or succeeded == len(download_queue):
                    print(f"  -> Downloaded {succeeded}/{len(download_queue)} files ({total_bytes / 1e6:.1f} MB)...", flush=True)
            else:
                failed += 1
                print(f"  ⚠ Failed to download: {fname}", flush=True)

    elapsed = time.time() - start_time
    print(f"[Phase 1 Downloader] Completed in {elapsed:.1f}s | Success: {succeeded} | Failed: {failed}")
    return {"total": len(download_queue), "downloaded": succeeded, "failed": failed}

def verify_dataset_completeness() -> bool:
    """Verifies that all 53 patient header files and data files exist."""
    missing = []
    for pid in PATIENT_IDS:
        if not (DATA_DIR / f"{pid}.hea").exists() or not (DATA_DIR / f"{pid}.dat").exists():
            missing.append(pid)
    if missing:
        print(f"[Phase 1 Downloader] Missing records: {missing}")
        return False
    print(f"[Phase 1 Downloader] Dataset verification: All 53 patient records verified in {DATA_DIR}.")
    return True
