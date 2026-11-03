# Remontada

Team Remontada's entry for **More than a Hack 2027, Phase 1**: label every frame of FC Barcelona vs Athletic Club (199,319 frames) with who has the ball (`H`, `A`, `D` or `X`), scored on spellF1.

Design: [docs/superpowers/specs/2026-10-30-possession-pipeline-design.md](docs/superpowers/specs/2026-10-30-possession-pipeline-design.md)

## Setup

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m pytest
```

Tests marked `realdata` need the data below and are skipped without it.

## Data

The match data is licensed for the competition only and is **never** in this repository.

- Put the official bundle in a folder outside the repo (`tactical/` and `broadcast/` inside it) and point `MTAH_DATA_DIR` at it.
- IDSSE goes in `$MTAH_DATA_DIR/idsse` (or `REMONTADA_IDSSE_DIR`). Fetch it with `python scripts/download_idsse.py` (about 2.5 GB).
- **Never paste the match data, or any extract of it, into a hosted or cloud AI tool.** The competition terms forbid it.
- Keep test tracebacks short (`--tb=line`, set in `pyproject.toml`): a long traceback from a failed real-data test could print match data. Do not override it when an AI tool can read your terminal.

## Running it

```bash
python scripts/scoreboard.py                                # the official baseline on IDSSE
python scripts/scoreboard.py --method hmm                   # our decoder on IDSSE
python scripts/scoreboard.py --method hmm --no-ball-hints   # the same without ball-dead hints
python scripts/tune_rough.py                                # rough settings grid (plan 2)
python scripts/tune_rough.py --write-counts                 # relearn config/evidence_idsse.json
python -m remontada.make_submission --method hmm --out outputs/remontada_hmm.csv
```

The decoder's settings are in `src/remontada/config/hmm_v1.json`.

## Credits

- The official scorer and public ground truth in `src/remontada/vendor/` come from [metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027) (MIT).
- IDSSE: Bassek, M. et al. "An integrated dataset of synchronized spatiotemporal and event data in elite soccer." *Scientific Data* 12, 195 (2025). doi:10.1038/s41597-025-04505-y. Licensed CC BY 4.0, mirrored at [pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data). `src/remontada/config/evidence_idsse.json` holds counts derived from it (CC BY 4.0).
