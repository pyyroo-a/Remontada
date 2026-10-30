"""Write the frame,label CSV. Same bytes on every computer: plain \\n line endings."""
import hashlib
from pathlib import Path

from remontada.vendor.possession_metrics import LABELS


def write_csv(codes, path):
    """Write label codes as a frame,label CSV and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["frame,label\n"]
    lines += ["{},{}\n".format(frame, LABELS[code]) for frame, code in enumerate(codes)]
    path.write_bytes("".join(lines).encode("ascii"))
    return path


def sha256(path):
    """SHA-256 of a file, as a hex string."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
