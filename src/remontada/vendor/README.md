# Vendored files

Copied byte for byte from [metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027) at commit `f8f2629ff1e0985f81d7e59d8413678514ee1530`, under that repository's MIT licence (`LICENSE-mtah2027`).

- `possession_metrics.py`: the official metric library (`evaluation/possession_metrics.py`). Every score we report comes from it.
- `BAR-ATH_possession_gt_public.csv`: the public 5-minute ground truth (`evaluation/gt/`). Kept only so the vendored files can be fingerprinted and the scorer checked on itself; never used to score our output (rules clarification, 4 Nov 2026).

Do not edit these files. `tests/test_vendor.py` checks their SHA-256.
