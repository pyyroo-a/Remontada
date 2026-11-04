# Plan 3: Better Scores on Messy Tracking, Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** raise the honest medium and heavy practice spellF1 as far as the evidence supports, without overfitting, and freeze one setting (`hmm_v2.json` plus `scorer_v2.json` if kept) for the submission.

**Architecture:** first make the practice mess fairer (degrade v3) and fix the plan 2 review bugs; then build one honest measuring harness (nested leave-one-match-out, two seeds, two hiding styles); then try four improvements in order (wider search, gap filling, touch-aware switching, a learned scorer), each kept only if it passes the acceptance rule; then freeze.

**Tech Stack:** Python 3.11, numpy 2.4.6, pandas 3.1.0, pytest 8.4.2. No new dependency: the learned scorer is a regularised logistic regression written in numpy, its weights saved as JSON (the repo's `.gitignore` blocks `*.pkl` on purpose).

**Spec:** `docs/superpowers/specs/2026-11-04-plan-3-better-scores-design.md` (approved by Ashen, 4 Nov 2026). Background: `docs/superpowers/specs/2026-10-30-possession-pipeline-design.md`, `docs/superpowers/plans/2026-11-03-plan-2-clues.md`.

## Global Constraints

- Never open, print or read BAR-ATH files under `D:\hackathon\more than a hack\data\tactical` or `data\broadcast`, `data\README.md`, or anything in `outputs\`. Runs on BAR-ATH print only "rows N valid" and sha256 (rules clarification, 4 Nov 2026). Read only the last line of any real-data pytest run.
- Every setting, threshold and trained weight comes from the 7 IDSSE matches (CC BY 4.0). Nothing is checked or tuned against the public 5 minutes.
- `MTAH_DATA_DIR=D:/hackathon/more than a hack/data` for real-data runs; the venv is `Remontada\.venv`. Long runs go in the background.
- No overfitting (Ashen, 4 Nov: "make it as good as we can, but not overfit"): every grid, bin count, regularisation strength and iteration count is fixed in this plan before any score is seen; nothing is added to a grid after seeing results.
- Honest numbers: every reported practice score is nested: learned tables and models never see the scored match, and settings for match m are chosen on the other six (spec section 5).
- Acceptance rule: keep a step only if the nested medium-and-heavy mean (both hiding styles, both seeds) rises by at least `ACCEPT_BAR` (0.01, or twice the seed spread if that spread is over 0.005) and no level falls by more than 0.04.
- Ashen's 3 Nov decision 4 holds: learned parts cover only who has the ball (live frames) and players slowing; ball out and ball still stay fixed-rule hints and are never model inputs.
- The submission stays byte-identical on two runs. Emission scores are rounded to 6 decimals before Viterbi so tiny float differences between machines cannot flip a decision.
- Never commit or push; never add Co-Authored-By lines. After each task give Ashen the files to tick, a Summary and a Description, and wait for "committed".
- No em or en dashes in code, comments, docs or notes. Plain words; new terms go in the vault's Plain words guide.

## Review Focus

1. **Gap filling across a boundary:** a ball gap that touches a graded-run edge or the half-time separator, or is longer than the limit, must stay unseen; filled frames must never create a touch. Test in Task 6.
2. **A touch discount larger than the switch cost:** a switch must never become cheaper than staying (cost floored at 0), or labels flicker. Test in Task 7.
3. **Nested selection ties:** when two settings score the same on the other six matches, the choice must be the first point in grid order, so reruns give the same answer. Test in Task 4.
4. **The learned scorer on a mostly missing ball:** NaN clues must not produce NaN scores; the decoded path stays valid H/A/D. Test in Task 8.
5. **Rebuilding the submission:** two runs of `make_submission --method hmm` with `hmm_v2.json` (and the scorer file) give the same sha256. Test in Task 9.

---

### Task 1: Degrade v3: ball wobble and two hiding styles

**Files:**
- Modify: `src/remontada/degrade.py`, `tests/test_degrade.py`

**Interfaces:**
- Produces: `DEGRADE_VERSION = 3`; `HIDING_STYLES = ("hard", "random")`; `degrade(table, level, seed, hiding="hard")`. `"random"` places every hidden stretch uniformly (no hard-frame phase). The ball gets the same slow-drift wobble as players, `level.position_noise_m`, on seen frames only (NaN stays NaN). Order of random draws: hide ball, player wobble, ball wobble, track breaks.

- [ ] **Step 1: Write the failing tests**

```python
def test_ball_wobble_drifts_slowly(clean_table):
    out = degrade(clean_table, Level(0.0, 0.0, 0.5, 0.0), seed=0)
    noise = out.ball_xy - clean_table.ball_xy
    assert 0.4 < np.nanstd(noise) < 0.6
    assert np.nanstd(np.diff(noise, axis=0)) < 0.2

def test_ball_wobble_keeps_hidden_frames_hidden(clean_table):
    out = degrade(clean_table, LEVELS["heavy"], seed=0)
    assert np.isnan(out.ball_xy[:, 0]).sum() == int(0.60 * clean_table.n_frames)

def test_random_hiding_has_no_preference(clean_table):
    hard = (np.arange(N_FRAMES) // 250) % 5 == 0
    table = replace(clean_table, ball_z=np.where(hard, 2.0, 0.1))
    hidden = np.isnan(degrade(table, LEVELS["heavy"], seed=0, hiding="random").ball_xy[:, 0])
    assert abs(hidden[hard].mean() - hidden[~hard].mean()) < 0.05

def test_hard_hiding_is_the_default(clean_table):
    a = degrade(clean_table, LEVELS["medium"], seed=3)
    b = degrade(clean_table, LEVELS["medium"], seed=3, hiding="hard")
    np.testing.assert_array_equal(a.ball_xy, b.ball_xy)
```

- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/test_degrade.py -q`. Expected: the new tests FAIL (ball unchanged; unknown keyword `hiding`).
- [ ] **Step 3: Implement** in `degrade.py`; reuse `_wobble` for the ball; `ValueError` for an unknown hiding style.
- [ ] **Step 4: Run** the same command. Expected: all pass (old tests included).
- [ ] **Step 5: Hand-off.** Summary: `Degrade v3: ball wobble and a random hiding style`.

---

### Task 2: Plan 2 review fixes

**Files:**
- Modify: `src/remontada/frames.py`, `src/remontada/pipeline.py`, `src/remontada/touch.py`, `src/remontada/smart_tagging.py`
- Test: `tests/test_pipeline.py`, `tests/test_touch.py`, `tests/test_smart_tagging.py`, `tests/test_frames.py` (create if absent)

**Interfaces:**
- Produces: `frames.run_slice(table, start, stop) -> FrameTable` (rows `start:stop`, `graded_runs=[(0, stop - start)]`). `pipeline.features(table)` now computes every clue on each graded run separately and stitches them; frames outside runs get the "nothing known" cells (team cell for unseen nearness, no touch, no control; slow cell "unknown"; no ball clues). `touches` keeps a jump only if a player is within `TOUCH_RADIUS` of that frame before choosing the largest jump in the window (so a bigger jump far from everyone cannot hide a real touch). A malformed Smart Tagging time raises `ValueError("bad time in Smart Tagging instance <ID>")` from None, without the value.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline.py
def test_clues_do_not_cross_the_half_time_gap():
    # players stand still in both halves but swap ends; one NaN row separates the halves
    first = [[(-30.0, 0.0), (30.0, 0.0)]] * 100
    second = [[(30.0, 0.0), (-30.0, 0.0)]] * 100
    table = synthetic_table([(0.0, 0.0)] * 201, first + [[NAN_XY, NAN_XY]] + second, teams=[0, 1])
    table = replace(table, graded_runs=[(0, 100), (101, 201)])
    f = features(table)
    assert (f.slow[88:100] == 0).all() and (f.slow[101:113] == 0).all()  # still "< 0.5 m/s", not 46.9

def test_predict_with_the_ball_missing_most_of_a_half():
    table, labels = _two_halves_of_possession()
    ball = table.ball_xy.copy(); ball[20:380] = np.nan
    codes = predict(replace(table, ball_xy=ball), count(table, labels), Settings(0.2, Costs(8, 8, 8), True, False))
    assert set(codes.tolist()) <= {H, A, D}

def test_a_pass_flying_1m_from_an_opponent_stays_home():
    ball = [(0.0, 0.0)] * 10 + [(0.5 * (t - 10), 0.0) for t in range(10, 51)] + [(20.0, 0.0)] * 20
    players = [[(-0.5, 0.0), (20.5, 0.0), (10.0, 1.0)]] * len(ball)
    table = synthetic_table(ball, players, teams=[0, 0, 1])
    t = touches(table)
    assert not (t == 1).any()
    labels = np.array([H] * len(ball), dtype=np.int8)
    codes = predict(table, count(table, labels), Settings(0.4, Costs(4, 32, 32), True, False))
    assert (codes[10:51] == H).all()

# tests/test_touch.py
def test_a_far_glitch_does_not_hide_a_real_touch():
    # real bounce off a home player at frame 20; a bigger jump at frame 21 far from everyone
    ...  # build as test_bounce_off_a_home_player_is_a_home_touch, then add a 3 m spike at frame 22 with no player near
    assert t[20] == 0

# tests/test_smart_tagging.py
def test_bad_time_does_not_show_its_value(tmp_path):
    (tmp_path / "t.xml").write_text(SAMPLE.replace("447.38", "44x.38"), encoding="utf-8")
    with pytest.raises(ValueError) as err:
        parse_smart_tagging(tmp_path / "t.xml")
    assert "44x" not in str(err.value) and "instance 1" in str(err.value)
```

- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/test_pipeline.py tests/test_touch.py tests/test_smart_tagging.py -q`. Expected: the five new tests FAIL.
- [ ] **Step 3: Implement** the four fixes as described in Interfaces.
- [ ] **Step 4: Run** the full fast suite `.venv/Scripts/python.exe -m pytest -q`. Expected: all pass.
- [ ] **Step 5: Hand-off.** Summary: `Fix plan 2 review minors: clues per half, far glitches, tag errors`.

---

### Task 3: Clue settings and Settings v2

**Files:**
- Create: `src/remontada/clues.py`
- Modify: `src/remontada/touch.py`, `src/remontada/dead_ball.py`, `src/remontada/evidence.py`, `src/remontada/pipeline.py`
- Test: `tests/test_clues.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces:
  - `ClueSettings` frozen dataclass `(touch_dv=3.0, touch_radius=1.5, control_radius=1.5, control_min_frames=10, carry_frames=75, out_margin_m=1.0, gap_fill_frames=0)`; `DEFAULT_CLUES = ClueSettings()`.
  - `touches(table, dv=TOUCH_DV, radius=TOUCH_RADIUS, ignore=None)`, `control(table, radius=CONTROL_RADIUS, min_frames=CONTROL_MIN_FRAMES)`; evidence cell functions and `pipeline.features(table, clues=DEFAULT_CLUES)` pass them through. `ignore` (bool mask, Task 6) marks frames a touch window may not include.
  - `Settings` gains `clues=DEFAULT_CLUES`, `touch_discount=0.0`, `touch_window=0`, `scorer="tally"`; `load_settings` reads them when present (hmm_v1.json still loads unchanged) and accepts a nested `"clues"` object.

- [ ] **Step 1: Write the failing tests**

```python
def test_default_clues_change_nothing():
    table, labels = _two_halves_of_possession()
    a, b = features(table), features(table, DEFAULT_CLUES)
    np.testing.assert_array_equal(a.team, b.team)

def test_clue_settings_reach_the_cells():
    table, _ = _two_halves_of_possession()
    near = features(table, replace(DEFAULT_CLUES, control_radius=0.1))
    assert not np.array_equal(near.team, features(table).team)

def test_hmm_v1_still_loads_with_defaults():
    s = load_settings(CONFIG_DIR / "hmm_v1.json")
    assert s.clues == DEFAULT_CLUES and s.touch_discount == 0.0 and s.scorer == "tally"

def test_settings_round_trip_with_clues(tmp_path):
    ...  # write {"temperature": 1, "switch": 10, ..., "clues": {"carry_frames": 50}} and load it
    assert load_settings(path).clues.carry_frames == 50
```

- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/test_clues.py tests/test_pipeline.py -q`. Expected: FAIL (no `clues` module).
- [ ] **Step 3: Implement.** Module constants stay as the defaults; nothing else changes behaviour.
- [ ] **Step 4: Run** the full fast suite. Expected: all pass.
- [ ] **Step 5: Hand-off.** Summary: `Make clue settings searchable`.

---

### Task 4: The honest practice harness

**Files:**
- Create: `src/remontada/practice.py`, `scripts/plan3.py`, `results/plan3/` (JSON results, IDSSE only, committed)
- Test: `tests/test_practice.py`

**Interfaces:**
- Consumes: `degrade(..., hiding=)` (Task 1), `features(table, clues)` (Task 3), `Settings` (Task 3), `decode_runs`, `score`.
- Produces (`practice.py`):
  - `SEEDS = (0, 1)`; a case is `(match_id, level_name, hiding, seed)`; `MEDIUM_HEAVY = ("medium", "heavy")`.
  - `case_features(clues) -> (feats: dict[case, Features], labels: dict[match_id, np.ndarray])`, degrading each case on the fly (never holding all tables).
  - `score_point(feats, labels, settings) -> dict[case, tuple[float, float]]` (spellF1, macroF1), tally counts left out per match, decoding in batches of one hiding and seed at a time.
  - `level_means(case_scores, matches=None) -> dict[level, float]` (spellF1 means over the given matches, both hidings and seeds).
  - `objective(case_scores, matches=None) -> float` = mean of `level_means` over `MEDIUM_HEAVY`.
  - `nested_estimate(results: dict[str, dict[case, tuple]]) -> (dict[level, float], dict[match, str])`: for each match m, pick the point with the best `objective` on the other six (ties: first key in insertion order), and average its scores on m.
  - `accepts(candidate: dict[level, float], current: dict[level, float], bar: float) -> bool`.
  - `seed_spread(case_scores) -> float` = |objective on seed 0 minus objective on seed 1|.
  - `save_results(path, results)` / `load_results(path)` as JSON (cases as `"match|level|hiding|seed"` strings).
- `scripts/plan3.py start`: scores the baseline and the Plan 2 setting (`hmm_v1.json`, tags off) on all 112 cases, prints both level tables, the seed spread and the resulting `ACCEPT_BAR`, and writes `results/plan3/start.json`.

- [ ] **Step 1: Write the failing tests** (pure functions on hand-made numbers)

```python
def _cases(matches, value_by_match):
    return {(m, lv, h, s): (value_by_match[m], 0.0) for m in matches for lv in LEVEL_NAMES
            for h in ("hard", "random") for s in (0, 1)}

def test_nested_estimate_picks_on_the_other_matches():
    ms = ["m1", "m2", "m3"]
    results = {"p": _cases(ms, {"m1": 0.9, "m2": 0.1, "m3": 0.1}),
               "q": _cases(ms, {"m1": 0.2, "m2": 0.3, "m3": 0.3})}
    levels, chosen = nested_estimate(results)
    assert chosen == {"m1": "q", "m2": "q", "m3": "q"}  # q wins on every set of two other matches
    assert abs(levels["heavy"] - (0.2 + 0.3 + 0.3) / 3) < 1e-12

def test_nested_ties_take_the_first_point():
    ms = ["m1", "m2"]
    results = {"p": _cases(ms, {"m1": 0.5, "m2": 0.5}), "q": _cases(ms, {"m1": 0.5, "m2": 0.5})}
    assert set(nested_estimate(results)[1].values()) == {"p"}

def test_accepts_needs_the_bar_and_the_guard():
    cur = {"none": 0.6, "light": 0.5, "medium": 0.4, "heavy": 0.2}
    assert accepts({**cur, "medium": 0.42, "heavy": 0.22}, cur, 0.01)
    assert not accepts({**cur, "medium": 0.405, "heavy": 0.205}, cur, 0.01)
    assert not accepts({**cur, "none": 0.55, "medium": 0.43, "heavy": 0.23}, cur, 0.01)

def test_results_round_trip(tmp_path): ...
```

- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/test_practice.py -q`. Expected: FAIL (no module).
- [ ] **Step 3: Implement** `practice.py` and `scripts/plan3.py start`.
- [ ] **Step 4: Run** the tests (pass), the full fast suite (pass), then `scripts/plan3.py start` in the background with `MTAH_DATA_DIR` set. Record in the vault Scoreboard a "Degrade v3 starting point" section: baseline and Plan 2 per level (nested for Plan 2 is the plain leave-one-out, as it has one setting), the seed spread and `ACCEPT_BAR`. These are the numbers Plan 3 must beat.
- [ ] **Step 5: Hand-off.** Summary: `Add the honest practice harness and re-score on degrade v3`.

---

### Task 5: Step A, the wider search

**Files:**
- Modify: `scripts/plan3.py` (subcommands `search-costs`, `search-clues`)
- Test: `tests/test_practice.py` (grid builders)

**Interfaces:**
- Consumes: Task 4 harness.
- Produces: `practice.cost_grid() -> list[Settings]`: temperature 1, `switch` in {2.5, 5, 10, 20, 40}, `to_dead` in {20, 40, 80, 160, 320}, `from_dead` in {20, 40, 80, 160, 320} (125 points, nats; plan 2's best is switch 10, dead 80 at temperature 1, the grid centre). `practice.clue_steps() -> list[tuple[str, list]]`: coordinate search over `touch_dv` {2, 3, 4.5}, `touch_radius` {1.0, 1.5, 2.0}, `control_radius` {1.0, 1.5, 2.0}, `control_min_frames` {5, 10, 20}, `carry_frames` {25, 50, 75, 125}, `out_margin_m` {0.5, 1.0, 1.5}, in that order, each value tried with the others at their current best and costs at the `search-costs` winner. Points are named by their settings so results files are readable.

- [ ] **Step 1: Write the failing tests:** `test_cost_grid_is_centred_on_plan_2` (125 points, contains switch 10 / dead 80 / dead 80, all temperature 1); `test_clue_steps_start_from_defaults` (each value list contains the `DEFAULT_CLUES` value).
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement**; each subcommand writes `results/plan3/search-costs.json` / `search-clues.json`, prints the nested level table against the current best, and prints KEEP or DROP from `accepts`.
- [ ] **Step 4: Run** the tests (pass), then both searches in the background (costs first). Record both nested tables and the verdict in the vault Scoreboard.
- [ ] **Step 5: Hand-off.** Summary: `Step A: wider cost and clue search, nested`.

---

### Task 6: Step B, filling short ball gaps

**Files:**
- Create: `src/remontada/gaps.py`, `tests/test_gaps.py`
- Modify: `src/remontada/pipeline.py`, `scripts/plan3.py` (`search-gaps`)

**Interfaces:**
- Produces: `gaps.fill_ball_gaps(ball_xy, runs, max_frames) -> (filled_xy, filled_mask)`: a gap is a stretch of NaN ball frames with a seen frame directly before and after it inside the same graded run; gaps of length 1 to `max_frames` are filled by a straight line between those two frames; `filled_mask` marks filled frames. `features(table, clues)` fills when `clues.gap_fill_frames > 0` and passes `filled_mask` as `touches(..., ignore=)`. Search `gap_fill_frames` in {0, 12, 25, 50, 75}.

- [ ] **Step 1: Write the failing tests**

```python
def test_fills_a_short_gap_in_a_straight_line():
    xy = np.array([[0, 0], [np.nan, np.nan], [np.nan, np.nan], [3, 0]], float)
    filled, mask = fill_ball_gaps(xy, [(0, 4)], 5)
    np.testing.assert_allclose(filled[:, 0], [0, 1, 2, 3]) ; assert mask.tolist() == [False, True, True, False]

def test_long_gaps_and_open_ends_stay_unseen():
    xy = np.array([[np.nan, np.nan], [0, 0]] + [[np.nan, np.nan]] * 6 + [[6, 0], [np.nan, np.nan]], float)
    filled, mask = fill_ball_gaps(xy, [(0, 10)], 5)
    assert not mask.any()

def test_gaps_never_bridge_two_runs():
    xy = np.array([[0, 0], [np.nan, np.nan], [2, 0]], float)
    assert not fill_ball_gaps(xy, [(0, 2), (2, 3)], 5)[1].any()

def test_filled_frames_never_make_a_touch():
    # ball rolls, vanishes 10 frames, reappears on a different line; no player near the joins
    ...
    assert (touches(table, ignore=mask) == -1).all()
```

- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/test_gaps.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** (vectorised: find gap starts and stops with `np.diff` on the NaN mask, per run).
- [ ] **Step 4: Run** the tests and full fast suite (pass), then `search-gaps` in the background. Record the nested table and KEEP or DROP.
- [ ] **Step 5: Hand-off.** Summary: `Step B: fill short ball gaps`.

---

### Task 7: Step C, easier switches after a touch

**Files:**
- Modify: `src/remontada/hmm.py`, `src/remontada/pipeline.py`, `src/remontada/practice.py`, `scripts/plan3.py` (`search-touch`)
- Test: `tests/test_hmm.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces: `viterbi(scores, costs, entry_bonus=None)` and `decode_runs(scores, runs, costs, entry_bonus=None)`: `entry_bonus` has the shape of `scores`; moving from state j to a different state k at frame t costs `max(cost[j, k] - entry_bonus[t, k], 0)`; staying costs 0. `pipeline.touch_entry_bonus(touch_codes, discount, window) -> np.ndarray (n, 3)`: `discount` on column k for frames within `window` of a touch by team k (no stacking), 0 elsewhere, column D always 0. Used when `settings.touch_discount > 0`. Search `touch_discount` as a share of `switch` in {0, 0.25, 0.5, 0.75, 1.0} times `window` in {0, 5, 12, 25}.

- [ ] **Step 1: Write the failing tests**

```python
def test_entry_bonus_matches_brute_force():
    ...  # 300 random short sequences with random bonus, as the plan 2 brute-force check, assert 0 mismatches

def test_a_bonus_bigger_than_the_cost_never_makes_switching_free_to_flicker():
    s = np.zeros((40, 3)); s[:, H] = 1
    bonus = np.zeros((40, 3)); bonus[:, A] = 100.0
    assert (viterbi(s, Costs(5, 5, 5), entry_bonus=bonus) == H).all()

def test_switch_lands_on_the_touch():
    # H evidence fades over frames 10..30 and A grows; a touch by A at frame 25
    ...
    assert first_a_frame(codes_with_bonus) == 25
```

- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/test_hmm.py tests/test_pipeline.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement**; with `entry_bonus=None` the decoder is unchanged (existing tests still pass).
- [ ] **Step 4: Run** tests and full fast suite (pass), then `search-touch` in the background. Record and KEEP or DROP.
- [ ] **Step 5: Hand-off.** Summary: `Step C: cheaper switches next to a touch`.

---

### Task 8: Step D, the learned scorer

**Files:**
- Create: `src/remontada/scorer.py`, `tests/test_scorer.py`
- Modify: `src/remontada/pipeline.py`, `src/remontada/practice.py`, `scripts/plan3.py` (`search-scorer`)

**Interfaces:**
- Produces:
  - `FRAME_CLUES`: names of the per-frame inputs, all computed from positions the same way on both datasets: nearness margin (clipped to plus or minus 20 m, NaN when unseen), nearest distance, last touch team (+1 home, -1 away, 0 none) and its age in s, control team and its age in s, ball speed, ball seen (0/1), seconds since the ball was last seen, ball filled (0/1), |ball x|, |ball y|. Never ball out or ball still.
  - `frame_clues(table, clues) -> np.ndarray float64 (n, len(FRAME_CLUES))`; `mirror(X)` flips the sign of the team-signed columns exactly (margin, last touch team, control team).
  - `Scorer` (bin edges per clue, one-hot weights, intercept) with `fit(X, y, l2=1.0, iterations=25) -> Scorer` (Newton steps, fixed in advance, deterministic), `p_home(X) -> np.ndarray`, `to_json` / `from_json`. Each clue is cut into fixed quantile bins (8 per clue, plus a NaN bin) learned from the training data; plus one interaction block: nearness bin x last touch team x control team (the plan 2 tally's cells).
  - Training data: IDSSE live frames only (labels H or A), every 5th frame of each training case, plus its mirror with H and A swapped. Leave one match out for evaluation (7 scorers); the final scorer is trained on all 7.
  - Scores: H = log(2 p), A = log(2 (1 - p)), team part of D = 0 (the average of the two likelihoods is 1), with p clipped to [1e-4, 1 - 1e-4]; the slow part and ball hints as before. `hint_weight` for the scorer = median of |H minus A| over its training live frames.
  - `Settings.scorer = "model"` switches `features()` and `predict()` to it; the file is `config/scorer_v2.json`.

- [ ] **Step 1: Write the failing tests**

```python
def test_learns_a_clear_rule():
    rng = np.random.default_rng(0); X = rng.normal(size=(4000, len(FRAME_CLUES))); y = (X[:, 0] > 0).astype(int)
    p = Scorer.fit(X, y).p_home(X)
    assert ((p > 0.5) == y.astype(bool)).mean() > 0.9

def test_mirror_is_exact():
    ...  # p_home(mirror(X)) == 1 - p_home(X) for a scorer trained on mirrored data, within 1e-9

def test_nan_clues_give_finite_scores():
    X = np.full((10, len(FRAME_CLUES)), np.nan)
    assert np.isfinite(Scorer.fit(train_X, train_y).p_home(X)).all()

def test_scorer_decodes_a_mostly_missing_ball():
    # tests/test_pipeline.py: the synthetic two-halves match, ball NaN on frames 20..380, scorer settings
    assert set(predict(table_missing, counts, scorer_settings, scorer=scorer).tolist()) <= {H, A, D}

def test_training_twice_gives_identical_weights(): ...
def test_json_round_trip(tmp_path): ...
```

- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/test_scorer.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement**; keep memory bounded (training uses the subsample; full clue matrices are made one case at a time when scoring).
- [ ] **Step 4: Run** tests and full fast suite (pass), then `search-scorer` in the background (scorer in place of the tally, costs re-searched on a 3 x 3 x 3 grid around the current best). Record and KEEP or DROP. If kept, add the clue weights that matter most to the vault for the technical report.
- [ ] **Step 5: Hand-off.** Summary: `Step D: a learned scorer for who has the ball`.

---

### Task 9: Step E, final re-tune and freeze

**Files:**
- Create: `src/remontada/config/hmm_v2.json`, `src/remontada/config/scorer_v2.json` (only if Task 8 kept it)
- Modify: `src/remontada/pipeline.py` (`SETTINGS_PATH` to `hmm_v2.json`), `scripts/plan3.py` (`search-final`, `freeze`), `scripts/scoreboard.py` (reads v2), `tests/test_pipeline.py`, `tests/test_make_submission.py`, `README.md`

**Interfaces:**
- Produces: `search-final` re-searches the costs (and the touch discount and window if Task 7 was kept) on the final version with a 3-point grid per setting around the current best; `freeze` writes the winner on all 7 matches to `hmm_v2.json` (tag hints false) and, if used, the scorer trained on all 7 matches to `scorer_v2.json`, plus `evidence_idsse.json` relearned with the final clue settings.

- [ ] **Step 1: Write the failing tests:** `test_shipped_settings_are_v2` (`load_settings()` reads `hmm_v2.json`, temperature 1, tag hints off); the realdata `test_hmm_submission_is_valid_and_deterministic` keeps passing with v2.
- [ ] **Step 2: Run** them. Expected: FAIL (no v2 file).
- [ ] **Step 3: Run** `search-final` in the background, then `freeze`.
- [ ] **Step 4: Run** the full suite with `MTAH_DATA_DIR` (last line only) and `make_submission --method hmm --out outputs/remontada_hmm.csv` twice (validity and sha256 only; the two sha256 must match). Record in the vault: the final nested table against the degrade v3 starting point and the baseline, every step tried with KEEP or DROP and its numbers. Republish the team page with the new practice scores.
- [ ] **Step 5: Hand-off.** Summary: `Freeze plan 3 settings (hmm_v2)`.

---

### Task 10: Fresh review, fixes, notes, merge

- [ ] **Step 1:** One fresh reviewer (Opus) reads the spec, this plan and `git diff main...plan-3-better-scores`, with the Review Focus list and the data rule in its brief. It must not run anything that touches BAR-ATH.
- [ ] **Step 2:** Fix every Critical and Important finding test-first; re-run the affected searches only if decoder or clue code changed, and update the vault numbers if they moved.
- [ ] **Step 3:** Vault: `Plan 3 explained` (plain words, like Plan 2's), hub milestone and decisions, Scoreboard, Plain words guide, Open questions. Team page republished.
- [ ] **Step 4: Hand-off** of the review fixes; Ashen merges `plan-3-better-scores` into `main` and pushes.

## Not in this plan

Plan 4 (the technical document, README polish, the fresh-download rebuild check, going public, the form). Team shape, heatmaps and Smart Tagging hints (later rounds or never). Any optional experiment after Plan 4.
