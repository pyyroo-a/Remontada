import re
import subprocess

import pytest

BANNED = re.compile(r"(\.mp4|_atd\.(txt|xml)|_homography\.csv|_smart_tagging\.xml|DFL_.*\.xml|\.npz|\.npy|\.parquet|\.pkl|\.feather)$")


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


def test_no_data_files_are_tracked():
    tracked = git("ls-files").stdout.splitlines()
    assert tracked and not [name for name in tracked if BANNED.search(name)]


@pytest.mark.parametrize("name", [
    "frames.npz", "frames.npy", "frames.parquet", "cache.pkl", "table.feather",
    "BAR-ATH_frames.npy", "x.xml.part", "remontada.csv", "outputs/remontada.csv",
    "data/tactical/BAR-ATH_tactical.mp4", "docs/remontada.csv", "results/frames.csv",
])
def test_data_like_files_are_ignored(name):
    assert git("check-ignore", "-q", "--no-index", name).returncode == 0


def test_no_csv_outside_vendor_is_tracked():
    tracked = git("ls-files", "*.csv").stdout.splitlines()
    assert tracked == ["src/remontada/vendor/BAR-ATH_possession_gt_public.csv"]
