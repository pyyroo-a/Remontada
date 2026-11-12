# Remontada

Team Remontada's entry for **More than a Hack 2027, Phase 1**: label every frame of FC Barcelona vs Athletic Club (199,319 frames) with who has the ball (`H`, `A`, `D` or `X`), ranked on spellF1.

- **What we built and why:** the technical document, [docs/remontada.md](docs/remontada.md) (also as `docs/remontada.pdf`).
- **Every practice number, with how to rerun it:** [docs/results.md](docs/results.md).

## Requirements

- Python 3.11.
- The official competition data bundle (about 20 GB). It is licensed for the competition only and is **never** in this repository.
- Only for rerunning the practice searches: the public IDSSE dataset (about 2.5 GB). Making the submission does not need it.

## Setup

Windows (checked):

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
```

macOS and Linux (should work, not checked):

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
```

`requirements.txt` pins every package the environment installs, with exact versions.

## The data

- Put the official bundle in a folder **outside** the repository (with `tactical/` and `broadcast/` inside it) and point `MTAH_DATA_DIR` at that folder, for example `export MTAH_DATA_DIR="/path/to/data"` (or `$env:MTAH_DATA_DIR = "D:\path\to\data"` in PowerShell).
- IDSSE, only for the practice scripts, goes in `$MTAH_DATA_DIR/idsse` (or `REMONTADA_IDSSE_DIR`). Fetch it with `python scripts/download_idsse.py`.
- **Never paste the match data, or any extract of it, into a hosted or cloud AI tool.** The competition terms forbid it.
- Keep test tracebacks short (`--tb=line`, set in `pyproject.toml`): a long traceback from a failed real-data test could print match data.

## Reproduce the submission

With the virtual environment active and `MTAH_DATA_DIR` set:

```bash
python -m remontada.make_submission
```

It writes `outputs/remontada.csv` with our decoder (settings in `src/remontada/config/hmm_v2.json`, tally in `src/remontada/config/evidence_idsse.json`) and prints only that the file is valid and its fingerprint:

```
rows 199319  valid
sha256 b9c58cf97e4c9c9bf12c298db2760d23dc06717a370cd8e285f87f6ec9e330cd
```

That sha256 was checked on Windows 11 with Python 3.11 and the pinned packages, on two runs. The decoder has no randomness. The command runs the same checks as the official validator; to run the official one itself, clone [metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027) and run `python evaluation/validate_submission.py outputs/remontada.csv` from there (with the path to our file).

`--method baseline` writes the official baseline instead, and `--out` picks another path.

## Practice and tests

```bash
python -m pytest                              # fast tests; real-data tests skip without the data
python scripts/scoreboard.py --method hmm     # our decoder on the 7 IDSSE matches, made messy
python scripts/results_tables.py              # the tables on docs/results.md, from results/
python scripts/plan3.py start                 # Plan 3 practice from the start (see the script for every step)
```

The practice searches take several hours; their saved results are in `results/`.

## What is where

- `src/remontada/`: the pipeline (loaders, clues, the decoder, the submission).
- `src/remontada/config/`: `hmm_v2.json` (the shipped settings) and `evidence_idsse.json` (the tally learned on IDSSE). `hmm_v1.json` is Plan 2's, kept for the record.
- `src/remontada/vendor/`: the organisers' scorer and public files, copied byte for byte.
- `scripts/`: practice, scoring and helper scripts.
- `results/`: saved practice results (IDSSE only).
- `tests/`: the test suite.
- `docs/`: the technical document, the results page, and our design specs and plans (`docs/superpowers/`).

## Licence and credits

- Our code is under the MIT licence (`LICENSE`).
- `src/remontada/vendor/` comes from [metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027) at commit `f8f2629` (MIT, notice in `vendor/LICENSE-mtah2027`): the official metric library, and the public 5-minute ground truth, which is kept only so the vendored files can be fingerprinted and is **never** used to score or tune our output.
- IDSSE: Bassek, M. et al. "An integrated dataset of synchronized spatiotemporal and event data in elite soccer." *Scientific Data* 12, 195 (2025). doi:10.1038/s41597-025-04505-y. Licensed CC BY 4.0, mirrored at [pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data). `src/remontada/config/evidence_idsse.json` and `results/` hold numbers derived from it (CC BY 4.0).
- Built with an AI coding assistant that saw only our code and summary numbers, never the match data.
