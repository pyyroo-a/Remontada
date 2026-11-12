"""What shipping needs: a licence, a fully pinned environment and one command that writes our file."""
import re
from pathlib import Path

from remontada import make_submission

ROOT = Path(__file__).resolve().parents[1]
PINNED = re.compile(r"^([A-Za-z0-9_.\-]+)==[A-Za-z0-9_.\-]+(\s*;.*)?$")
MUST_PIN = {"numpy", "pandas", "pytest", "python-dateutil", "tzdata", "six", "packaging", "pluggy",
            "iniconfig", "pygments", "colorama"}


def test_licence_is_mit():
    licence = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in licence and "Team Remontada" in licence
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^license\s*=\s*("MIT"|\{\s*text\s*=\s*"MIT"\s*\})', pyproject, re.M)


def test_every_requirement_is_pinned():
    lines = [line.strip() for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    assert all(PINNED.match(line) for line in lines), [line for line in lines if not PINNED.match(line)]
    names = {PINNED.match(line).group(1).lower() for line in lines}
    assert MUST_PIN <= names, MUST_PIN - names


def test_default_method_is_hmm():
    args = make_submission.parse_args([])
    assert args.method == "hmm"
    assert Path(args.out) == Path("outputs/remontada.csv")


REFERENCE_SHA256 = "b9c58cf97e4c9c9bf12c298db2760d23dc06717a370cd8e285f87f6ec9e330cd"


def _readme():
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_has_the_one_command_and_the_fingerprint():
    readme = _readme()
    assert "python -m remontada.make_submission" in readme
    assert REFERENCE_SHA256 in readme


def test_readme_says_where_it_was_checked():
    readme = _readme()
    assert "Windows 11" in readme and "Python 3.11" in readme


def test_readme_has_no_dashes():
    readme = _readme()
    assert "—" not in readme and "–" not in readme
