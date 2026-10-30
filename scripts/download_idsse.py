"""Download the 7 IDSSE matches (about 3 GB, CC BY 4.0) into the IDSSE folder, outside git.

    python scripts/download_idsse.py

A file whose size already matches the server's is skipped; anything else (missing, empty or cut
short) is fetched again. Folder: REMONTADA_IDSSE_DIR, else $MTAH_DATA_DIR/idsse.
"""
import shutil
import sys
import urllib.request

from remontada.load_idsse import MATCH_IDS, file_names
from remontada.paths import idsse_dir

BASE_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"


def remote_size(url):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request) as response:
        return int(response.headers["Content-Length"])


def fetch(url, path):
    partial = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out)
    partial.replace(path)


def main():
    folder = idsse_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for match_id in MATCH_IDS:
        for name in file_names(match_id).values():
            url, path = BASE_URL + name, folder / name
            size = remote_size(url)
            if path.is_file() and path.stat().st_size == size:
                print("ok      ", name)
                continue
            print("fetching", name, "({:.0f} MB)".format(size / 1e6), flush=True)
            fetch(url, path)
            if path.stat().st_size != size:
                sys.exit("size mismatch after download: {}".format(name))
    print("done:", folder)


if __name__ == "__main__":
    main()
