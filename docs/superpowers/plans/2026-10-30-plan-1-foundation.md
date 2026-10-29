# Plan 1: Foundation (data, IDSSE, scoreboard) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tested Python package that loads BAR-ATH and IDSSE into one frame-table shape, degrades IDSSE at several levels, reproduces the official baseline byte for byte, scores anything with the official scorer, and writes a valid `remontada.csv` (baseline method for now).

**Architecture:** `src/remontada/` holds small single-purpose modules: paths, frame table, two loaders, degradation, baseline, scoreboard, submission. The official scorer is vendored unchanged. Scripts in `scripts/` download IDSSE, print IDSSE label statistics, and print the scoreboard. Tests use synthetic data; tests marked `realdata` run on the real files and assert only scalars.

**Tech Stack:** Python 3.11, numpy 2.4.6, pandas 3.1.0, pytest 8.4.2.

**Spec:** `docs/superpowers/specs/2026-10-30-possession-pipeline-design.md` (approved 30 Oct 2026). This is plan 1 of 3: plan 2 adds the cues and the HMM decoder; plan 3 adds tuning, the explanations report, the final submission and the technical document.

## Global Constraints

- **No BAR-ATH data to Claude:** never `cat`, `head`, `Read` or print any file under `$MTAH_DATA_DIR`, nor frame-level outputs made from it. Code run on BAR-ATH prints only summary numbers (counts, shapes, scores, hashes, pass or fail). `realdata` tests assert only scalars or hashes and run with `--tb=line`. IDSSE (CC BY 4.0) and the public challenge repo's files may be viewed freely.
- **Commits:** Claude never runs `git commit` or `git push`. Each task ends with a hand-off: the files to tick, a Summary line and a Description, then wait for Ashen to say "committed".
- **No em or en dashes** in code comments, docs or commit text.
- Python 3.11; `requirements.txt` pins exactly `numpy==2.4.6`, `pandas==3.1.0`, `pytest==8.4.2`. No other dependency in plan 1.
- Data location: BAR-ATH from env `MTAH_DATA_DIR` (folders `tactical/`, `broadcast/`); IDSSE from env `REMONTADA_IDSSE_DIR`, default `$MTAH_DATA_DIR/idsse`. Neither is ever inside the repo.
- BAR-ATH constants (DATA.md): 199,319 video frames; 25 fps; pitch 105.3 x 68.0 m; graded halves are video frames 11,185 to 90,651 and 110,808 to 190,197 inclusive (half-open runs `(11185, 90652)` and `(110808, 190198)`); ATD `frame_id - 1` = video frame; Smart Tagging seconds x 25, rounded.
- Frame table coordinates: metres, origin at the centre spot. BAR-ATH `x_m = (x - 0.5) * 105.3`, `y_m = (y - 0.5) * 68.0`. IDSSE is already in this form (pitch 105 x 68).
- Label codes everywhere follow the vendored scorer: `H=0, A=1, D=2, X=3`.
- The vendored scorer is never edited.

## Review Focus

1. **A frame where the ball is visible but no player is**: the baseline must give no answer for it (not crash, not pick slot 0). Test in Task 6.
2. **IDSSE players who come on or go off mid-half**: their slot is NaN outside their frames; frame indices never shift. Test in Task 3.
3. **`MTAH_DATA_DIR` unset or pointing at the wrong folder** (and paths with spaces, like `more than a hack`): a clear error naming the missing file, never a data dump. Test in Task 2.
4. **A truncated or empty IDSSE download**: the downloader fetches it again; the loader raises a clear error. Test in Task 3.
5. **Degradation at level "none"** returns a table equal to the input, and **"heavy"** never hides the ball on every frame. Test in Task 5.

---

### Task 1: Repo skeleton and vendored scorer

**Files:**
- Create: `requirements.txt`, `pyproject.toml` (package `remontada` under `src/`, pytest config with `testpaths = ["tests"]`, marker `realdata`), `.gitignore`, `README.md`
- Create: `src/remontada/__init__.py`, `src/remontada/vendor/__init__.py`
- Create: `src/remontada/vendor/possession_metrics.py` (byte-identical copy of upstream `evaluation/possession_metrics.py` at commit `f8f2629`)
- Create: `src/remontada/vendor/BAR-ATH_possession_gt_public.csv` (byte-identical copy of upstream `evaluation/gt/BAR-ATH_possession_gt_public.csv`; covered by the upstream MIT licence)
- Create: `src/remontada/vendor/LICENSE-mtah2027` (upstream `LICENSE`, unchanged) and `src/remontada/vendor/README.md` (source repo URL, commit `f8f2629`, "do not edit")
- Test: `tests/test_vendor.py`

**Interfaces:**
- Produces: `from remontada.vendor import possession_metrics as pm`; `remontada.vendor.PUBLIC_GT_PATH: pathlib.Path`.

- [ ] **Step 1: Write the failing test**

```python
def test_vendored_scorer_is_unchanged():
    assert sha256_of(pm.__file__) == VENDORED_SHA  # constant recorded from upstream f8f2629

def test_public_gt_scores_itself_perfectly():
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT_PATH))
    result = pm.evaluate(gt, gt)
    assert len(gt) == 199319
    assert result["spell_f1_mean"] == 1.0
    assert result["n_gt_spells"] == 17
    assert pm.valid_runs(gt) == [(11185, 18685)]
```

- [ ] **Step 2: Run** `python -m pytest tests/test_vendor.py -v`. Expected: FAIL (module not found).
- [ ] **Step 3: Add the files.** Copy the two upstream files byte for byte and record their SHA-256 in the test. `.gitignore` covers `data/`, `idsse/`, `outputs/`, `*.mp4`, `*_atd.txt`, `*_homography.csv`, `*_smart_tagging.xml`, `DFL_*.xml`, `.venv/`, `__pycache__/`, `.pytest_cache/`. `README.md`: what the project is, setup (`py -3.11 -m venv .venv`, `pip install -r requirements.txt`, `pip install -e .`), where the data goes, the "no data in the repo, no data to cloud LLMs" rule, and the IDSSE citation (Bassek, M. et al., Scientific Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y, CC BY 4.0).
- [ ] **Step 4: Run** `python -m pytest tests/test_vendor.py -v`. Expected: 2 passed.
- [ ] **Step 5: Hand-off.** Files: everything above. Summary: `Set up package, pinned deps and vendored official scorer`.

---

### Task 2: Paths and the BAR-ATH frame table

**Files:**
- Create: `src/remontada/paths.py`, `src/remontada/frames.py`, `src/remontada/load_bar_ath.py`
- Test: `tests/test_load_bar_ath.py`, `tests/conftest.py` (shared synthetic builders, `realdata` skip when `MTAH_DATA_DIR` is unset or missing files)

**Interfaces:**
- Produces:
  - `paths.data_dir() -> Path` (from `MTAH_DATA_DIR`, raises `DataNotFound` naming the variable when unset); `paths.bar_ath_file(asset: str) -> Path` for `"atd_meta" | "atd_raw" | "smart_tagging"`, raising `DataNotFound` with the full expected path when missing; `paths.idsse_dir() -> Path`.
  - `frames.FrameTable` (frozen dataclass): `match_id: str`, `fps: int`, `pitch: tuple[float, float]`, `ball_xy: np.ndarray (n, 2) float64` (NaN when missing), `ball_z: np.ndarray (n,) float64` (NaN when unknown), `player_xy: np.ndarray (n, slots, 2) float64` (NaN when absent), `player_team: np.ndarray (slots,) int8` (0 home, 1 away), `graded_runs: list[tuple[int, int]]` (half-open), property `n_frames`.
  - `load_bar_ath(data_dir: Path | None = None) -> FrameTable`, `match_id="BAR-ATH"`, one row per video frame, `graded_runs = [(11185, 90652), (110808, 190198)]`.
  - In `tests/conftest.py`: `synthetic_table(ball: list[tuple[float, float]], players: list[list[tuple[float, float]]], teams: list[int] | None = None) -> FrameTable` (metres, NaN allowed, one graded run over all frames), `NAN_XY = (nan, nan)`, and `write_synthetic_bar_ath(tmp_path) -> Path` (writes a 3-frame, 2-slot bundle in the real file formats).

- [ ] **Step 1: Write the failing tests**

```python
def test_parses_synthetic_atd(tmp_path):  # 3 frames, 2 slots (1 home, 1 away), ball missing on frame 2
    table = load_bar_ath(write_synthetic_bar_ath(tmp_path))
    assert table.n_frames == 3
    assert table.player_team.tolist() == [0, 1]
    assert np.isnan(table.ball_xy[1]).all()
    assert table.ball_xy[0].tolist() == pytest.approx([(0.6 - 0.5) * 105.3, (0.25 - 0.5) * 68.0])

def test_missing_data_dir_names_the_variable(monkeypatch):
    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    with pytest.raises(DataNotFound, match="MTAH_DATA_DIR"):
        paths.data_dir()

def test_missing_file_names_the_path(tmp_path):  # folder with a space in its name
    with pytest.raises(DataNotFound, match="BAR-ATH_tactical_atd.txt"):
        load_bar_ath(tmp_path / "more than a hack")

@pytest.mark.realdata
def test_real_bar_ath_shape():
    table = load_bar_ath()
    assert table.n_frames == 199319
    assert table.player_xy.shape[1] == 46
    assert int((table.player_team == 0).sum()) == 24
    assert int((~np.isnan(table.ball_xy[:, 0])).sum()) == 113488
```

- [ ] **Step 2: Run** `python -m pytest tests/test_load_bar_ath.py -v --tb=line`. Expected: FAIL (import error).
- [ ] **Step 3: Implement.** Raw ATD line format (DATA.md, `starter/baseline.py`): `<frame_id>:<slot>;<slot>;...:<ball>`, each group `x,y,speed` with `NaN` for missing; read x and y only. Slot teams from the metadata XML: the first `Team` element is home, each `Player` element's `teamId` in file order gives the slot's team. Stream the file line by line.
- [ ] **Step 4: Run** `python -m pytest tests/test_load_bar_ath.py -v --tb=line` with `MTAH_DATA_DIR` set. Expected: 4 passed.
- [ ] **Step 5: Hand-off.** Summary: `Load BAR-ATH tracking into a frame table`.

---

### Task 3: IDSSE download and loader

**Files:**
- Create: `scripts/download_idsse.py`, `src/remontada/load_idsse.py`
- Test: `tests/test_load_idsse.py`

**Interfaces:**
- Consumes: `paths.idsse_dir()`, `frames.FrameTable`.
- Produces:
  - `load_idsse.MATCH_IDS: tuple[str, ...] = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")` with competition `DFL-COM-000001` for the first two and `DFL-COM-000002` for the rest (from the Hugging Face file listing).
  - `load_idsse.load_idsse(match_id: str, idsse_dir: Path | None = None) -> tuple[FrameTable, np.ndarray]`: the table and int8 label codes. Rows are first half, then one separator row (all NaN, label `X`), then second half. `graded_runs` are the two halves. Label per frame: `BallStatus == "0"` gives D, else `BallPossession` 1 gives H and 2 gives A. Home is `HomeTeamId` from the match information file.
  - `scripts/download_idsse.py`: downloads `matchinformation`, `events_raw` and `positions_raw_observed` for all 7 matches from `https://huggingface.co/datasets/pysport/idsse-data/resolve/main/<file>` into `idsse_dir()`, skipping files whose size matches the remote `Content-Length`, re-fetching any that do not.

- [ ] **Step 1: Write the failing tests**

```python
def test_synthetic_idsse(tmp_path):  # 2 halves of 4 frames; one home player only from frame 3 of half 1
    table, labels = load_idsse("TEST01", write_synthetic_idsse(tmp_path))
    assert table.n_frames == 4 + 1 + 4
    assert labels.tolist()[4] == pm.X
    assert table.graded_runs == [(0, 4), (5, 9)]
    assert np.isnan(table.player_xy[0:2, LATE_SLOT]).all()
    assert not np.isnan(table.player_xy[2, LATE_SLOT]).any()

def test_truncated_file_raises_clearly(tmp_path):
    with pytest.raises(ValueError, match="positions"):
        load_idsse("TEST01", write_truncated_idsse(tmp_path))

@pytest.mark.realdata
def test_real_idsse_koeln_bayern():  # values measured 29 Oct 2026
    table, labels = load_idsse("J03WMX")
    assert table.n_frames == 70708 + 1 + 75259
    counts = {c: int((labels == c).sum()) for c in (pm.H, pm.A, pm.D)}
    assert counts == {pm.H: 39111, pm.A: 46258, pm.D: 60598}
```

- [ ] **Step 2: Run** `python -m pytest tests/test_load_idsse.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** The positions file is about 420 MB: stream it line by line and read attributes with a regex (measured at about 4 s for one match), not a DOM parser. Each `<FrameSet>` has `GameSection` (`firstHalf` or `secondHalf`), `TeamId` (a club id, `referee` or `BALL`) and `PersonId`; each `<Frame>` has `N`, `X`, `Y` (and for the ball `Z`, `BallPossession`, `BallStatus`). Row index is `N` minus the half's first ball `N`. Skip the referee. One slot per `PersonId` of either club.
- [ ] **Step 4: Run** `python scripts/download_idsse.py`, then `python -m pytest tests/test_load_idsse.py -v`. Expected: 3 passed.
- [ ] **Step 5: Hand-off.** Summary: `Download and load the IDSSE matches with per-frame labels`.

---

### Task 4: IDSSE label statistics (the convention check)

**Files:**
- Create: `src/remontada/stats.py`, `scripts/idsse_label_stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: `load_idsse`, `pm.valid_runs`.
- Produces: `stats.possession_summary(codes: np.ndarray, runs: list[tuple[int, int]], fps: int = 25) -> dict` with keys `n_possessions`, `mean_s`, `median_s`, `n_under_1s`, `n_dead`, `mean_dead_s`, `alive_share`. Possessions are maximal H or A runs inside each run; D runs are stoppages; nothing crosses a run boundary.

- [ ] **Step 1: Write the failing test**

```python
def test_summary_counts_runs_inside_halves_only():
    codes = np.array([H, H, A, A, A, D, X, D, H, H], dtype=np.int8)
    s = possession_summary(codes, [(0, 6), (7, 10)], fps=1)
    assert s["n_possessions"] == 3
    assert s["n_dead"] == 2
    assert s["n_under_1s"] == 0
```

- [ ] **Step 2: Run** `python -m pytest tests/test_stats.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `possession_summary`, and the script printing one row per IDSSE match plus the mean over all 7.
- [ ] **Step 4: Run** the test (PASS), then `python scripts/idsse_label_stats.py`. Expected: 7 rows; J03WMX shows 511 possessions (250 + 261) and 103 dead stretches (41 + 62). Paste the table into the hand-off.
- [ ] **Step 5: Hand-off.** Summary: `Report possession statistics for all IDSSE matches`. Claude also adds the table to the Obsidian note "IDSSE dataset".

---

### Task 5: Degradation (light to heavy)

**Files:**
- Create: `src/remontada/degrade.py`
- Test: `tests/test_degrade.py`

**Interfaces:**
- Consumes: `FrameTable`.
- Produces: `degrade.Level` (frozen dataclass: `ball_hidden_share: float`, `ball_gap_mean_s: float`, `position_noise_m: float`, `track_breaks_per_min: float`); `degrade.LEVELS: dict[str, Level]` = `none (0, 0, 0, 0)`, `light (0.10, 0.5, 0.10, 0.5)`, `medium (0.30, 1.0, 0.25, 1.0)`, `heavy (0.60, 2.0, 0.50, 2.0)` per player; `degrade.degrade(table: FrameTable, level: Level, seed: int) -> FrameTable`. These levels span a wide range on purpose and are not taken from BAR-ATH (decided 30 Oct: sweep a range).

- [ ] **Step 1: Write the failing tests**

```python
def test_none_level_is_identity(clean_table):
    out = degrade(clean_table, LEVELS["none"], seed=0)
    np.testing.assert_array_equal(out.ball_xy, clean_table.ball_xy)
    np.testing.assert_array_equal(out.player_xy, clean_table.player_xy)

def test_ball_hidden_share_close_to_target(clean_table):  # 25,000 synthetic frames, ball always visible
    out = degrade(clean_table, LEVELS["medium"], seed=0)
    assert abs(np.isnan(out.ball_xy[:, 0]).mean() - 0.30) < 0.03

def test_heavy_never_hides_every_frame(clean_table):
    assert not np.isnan(degrade(clean_table, LEVELS["heavy"], seed=0).ball_xy[:, 0]).all()

def test_same_seed_same_output(clean_table):
    a, b = (degrade(clean_table, LEVELS["heavy"], seed=7) for _ in range(2))
    np.testing.assert_array_equal(a.player_xy, b.player_xy)

def test_track_breaks_keep_team_and_player_count(clean_table):
    out = degrade(clean_table, LEVELS["heavy"], seed=0)
    assert out.player_xy.shape[1] > clean_table.player_xy.shape[1]
    visible = lambda t, team: (~np.isnan(t.player_xy[:, t.player_team == team, 0])).sum(axis=1)
    for team in (0, 1):
        np.testing.assert_array_equal(visible(out, team), visible(clean_table, team))
```

- [ ] **Step 2: Run** `python -m pytest tests/test_degrade.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Ball gaps: hidden stretches with exponential lengths (mean `ball_gap_mean_s`) placed until the hidden share is reached; position noise: Gaussian on visible players; track breaks: at random frames a player's remaining frames move to a new slot of the same team. Use one `np.random.default_rng(seed)`.
- [ ] **Step 4: Run** `python -m pytest tests/test_degrade.py -v`. Expected: 5 passed.
- [ ] **Step 5: Hand-off.** Summary: `Degrade clean tracking at four levels for honest testing`.

---

### Task 6: Baseline on any frame table, reproduced byte for byte

**Files:**
- Create: `src/remontada/baseline.py`, `src/remontada/submission.py`
- Test: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `FrameTable`, `load_bar_ath`.
- Produces: `baseline.nearest_team(table: FrameTable) -> np.ndarray` (int8 per frame: 0, 1, or -1 when the ball or every player is missing); `baseline.smooth(codes: np.ndarray, window: int) -> np.ndarray` (label codes H, A, D: majority vote over plus or minus `window` frames, ties to H, D where no vote, as in upstream `smooth`); `submission.write_csv(codes: np.ndarray, path: Path) -> Path` (header `frame,label`, one `"{frame},{label}\n"` line per frame, `\n` endings); `submission.sha256(path: Path) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
def test_ball_without_players_gives_no_answer():
    table = synthetic_table(ball=[(0.0, 0.0)], players=[[NAN_XY, NAN_XY]])
    assert nearest_team(table).tolist() == [-1]

def test_nearest_player_team(): ...  # ball at origin, away player 1 m away, home player 3 m away -> [1]

@pytest.mark.realdata
@pytest.mark.parametrize("window,expected_sha", [
    (0,   "2a639a6b3c7816ab619d543cc870b021dca6945980eae8978325be0cc311fd63"),
    (25,  "57330f8bc13846751ceaf745f300f8bdb33cb92849b708d5350724eb7f843532"),
    (50,  "d444caa1aebb6fb58386033a5967c83302b8dd984ea91c86381bf208332702ff"),
    (125, "9b81f2027bd19224d900bdd7529de3fac078ce21160b5228201bbe86f94b7991"),
])
def test_reproduces_official_baselines(tmp_path, window, expected_sha):  # hashes of upstream evaluation/results/*.csv
    codes = smooth(nearest_team(load_bar_ath()), window)
    assert sha256(write_csv(codes, tmp_path / "b.csv")) == expected_sha
```

- [ ] **Step 2: Run** `python -m pytest tests/test_baseline.py -v --tb=line`. Expected: FAIL.
- [ ] **Step 3: Implement.** Squared distance in metres, first slot wins ties (as upstream's strict `<`), ignoring NaN slots. If a hash differs, use superpowers:systematic-debugging and compare only counts and hashes, never print rows; the centre-origin conversion changing float rounding is the first suspect.
- [ ] **Step 4: Run** the same command. Expected: 6 passed.
- [ ] **Step 5: Hand-off.** Summary: `Run the official baseline on any frame table, byte for byte`.

---

### Task 7: Scoreboard and the first valid submission

**Files:**
- Create: `src/remontada/scoreboard.py`, `src/remontada/make_submission.py`, `scripts/scoreboard.py`
- Test: `tests/test_scoreboard.py`, `tests/test_make_submission.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `scoreboard.score(gt: np.ndarray, pred: np.ndarray) -> dict` (keys `spell_f1`, `macro_f1`, `n_pred_spells`, `n_gt_spells`, from `pm.evaluate`); `scoreboard.mask_ungraded(codes, runs) -> np.ndarray` (X outside the runs); `make_submission.main(argv: list[str]) -> Path`, run as `python -m remontada.make_submission --method baseline --out outputs/remontada.csv` (writes the masked file, runs `pm.validate_submission` with 199,319 rows, prints only row count, label counts, possession count and the SHA-256); `scripts/scoreboard.py` prints baseline (window 50) scores for every IDSSE match at every level, then BAR-ATH on the public window.

- [ ] **Step 1: Write the failing tests**

```python
def test_mask_ungraded():
    out = mask_ungraded(np.array([H, H, A, A, D]), [(1, 3)])
    assert out.tolist() == [X, H, A, X, X]

@pytest.mark.realdata
def test_bar_ath_public_window_matches_readme():
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT_PATH))
    s = score(gt, smooth(nearest_team(load_bar_ath()), 50))
    assert round(s["spell_f1"], 4) == 0.3256
    assert round(s["macro_f1"], 4) == 0.6207

@pytest.mark.realdata
def test_submission_is_valid_and_deterministic(tmp_path):
    a, b = (main(["--method", "baseline", "--out", str(tmp_path / f"{i}.csv")]) for i in range(2))
    assert pm.validate_submission(pm.read_label_csv(a), 199319) is None
    assert sha256(a) == sha256(b)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_scoreboard.py tests/test_make_submission.py -v --tb=line`. Expected: FAIL.
- [ ] **Step 3: Implement** the three pieces. `make_submission` uses `mask_ungraded` with BAR-ATH's two graded halves.
- [ ] **Step 4: Run** the tests (PASS), the full suite `python -m pytest --tb=line` (all pass), then `python scripts/scoreboard.py`. Expected: 7 matches x 4 levels of IDSSE scores and the BAR-ATH line `spellF1 0.3256 macroF1 0.6207`. Also run the official `validate_submission.py` from the upstream clone on `outputs/remontada.csv`: `Valid.`
- [ ] **Step 5: Hand-off.** Summary: `Add the scoreboard and a valid baseline submission`. Claude adds the IDSSE baseline table to the Obsidian vault as a new milestone and note.

---

## After plan 1

Plan 2 (cues and the decoder, target 6 Nov) is written once this scoreboard exists, so its targets come from real IDSSE numbers. Plan 3 (tuning, explanations, final submission, technical document) follows.
