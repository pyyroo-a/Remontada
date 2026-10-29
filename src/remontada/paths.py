"""Where the data lives. Nothing here opens a file; it only builds and checks paths."""
import os
from pathlib import Path

BAR_ATH_FILES = {
    "atd_meta": "BAR-ATH_tactical_atd.xml",
    "atd_raw": "BAR-ATH_tactical_atd.txt",
    "smart_tagging": "BAR-ATH_tactical_smart_tagging.xml",
}


class DataNotFound(FileNotFoundError):
    """The licensed data is not where we expected it. The message names the path or variable."""


def data_dir():
    """Root of the official data bundle, from MTAH_DATA_DIR."""
    root = os.environ.get("MTAH_DATA_DIR")
    if not root:
        raise DataNotFound("MTAH_DATA_DIR is not set; point it at the folder holding tactical/")
    return Path(root)


def bar_ath_file(asset, root=None):
    """Path to one tactical BAR-ATH asset, or DataNotFound naming the full expected path."""
    return bar_ath_files([asset], root)[0]


def bar_ath_files(assets, root=None):
    """Paths to several tactical BAR-ATH assets, or one DataNotFound naming every missing one."""
    folder = Path(root or data_dir()) / "tactical"
    found = [folder / BAR_ATH_FILES[asset] for asset in assets]
    missing = [str(path) for path in found if not path.is_file()]
    if missing:
        raise DataNotFound("match data not found:\n  {}".format("\n  ".join(missing)))
    return found


def idsse_dir():
    """IDSSE download folder: REMONTADA_IDSSE_DIR, else $MTAH_DATA_DIR/idsse."""
    root = os.environ.get("REMONTADA_IDSSE_DIR")
    return Path(root) if root else data_dir() / "idsse"
