# Plan 3 design: better scores on messy tracking

Team Remontada, More than a Hack 2027, Phase 1. Written 4 Nov 2026 from the Plan 3 brainstorm with Ashen. Builds on the main design (`2026-10-30-possession-pipeline-design.md`) and Plan 2 (`docs/superpowers/plans/2026-11-03-plan-2-clues.md`, merged 4 Nov as pull request #2).

## 1. Why, and what success looks like

The goal is a **top 20 finish**, so the next round is reached. Phase 1 is ranked only on spellF1 (macroF1 breaks ties), with no leaderboard and no feedback before the deadline (17 Nov 2026). So Plan 3 exists to raise the score as far as it safely can.

Today (Plan 2, IDSSE, leave one match out, mean spellF1):

| Mess level | Baseline | Plan 2 |
|---|---|---|
| none | 0.3081 | 0.6650 |
| light | 0.3053 | 0.5891 |
| medium | 0.2684 | 0.4553 |
| heavy | 0.1743 | 0.2727 |

The lead shrinks as the data gets messier, and real automatic tracking is messy (the organisers' DATA.md says it is automatic and uncorrected). So **success for Plan 3 is a clearly higher score on the medium and heavy practice levels, without losing ground on clean data**, measured honestly.

Ashen's stance (his words, paraphrased): sure gains first; nothing that can eat several days and come back empty. Every change must pay off on the practice matches or be dropped.

## 2. Goals, non-goals, anti-goals

**Goals**
- Raise mean spellF1 on medium and heavy practice data, through the six steps in section 5.
- Make practice data a fairer stand-in for real tracking before tuning on it (section 4).
- Fix the plan 2 review's deferred minors that are real bugs or test gaps (section 5, step F).
- End with one frozen setting (config plus any trained weights) that `make_submission --method hmm` uses, ready for Plan 4.

**Non-goals (later, or never)**
- Team shape and heatmaps: kept for the later rounds (Ashen, 4 Nov).
- Smart Tagging hints: stay off; there is no fair way to test them on outside data.
- A ball detector on the video: too slow and too uncertain for the time left.
- The technical document, README, going public and the form: that is **Plan 4**, designed separately.
- X abstentions inside graded frames: an X counts as a miss in macroF1 and is filled with our last label in spellF1, so it cannot help.

**Anti-goals (would make Plan 3 a failure even if the code works)**
- Any tuning, fitting, checking or choosing on our match (BAR-ATH) or its public 5 minutes. The 4 Nov rules clarification allows tuning and training only on outside data.
- A change kept because it helps clean data while hurting messy data.
- Scores that rise because practice got easier rather than because the decoder got better.
- Anything that stops `make_submission` from rebuilding a byte-identical file.
- Any BAR-ATH-derived number reaching Claude or the notes, beyond "valid, identical twice, fingerprint".

## 3. Constraints (unchanged from Plans 1 and 2 unless stated)

- **Data rule:** Claude never opens BAR-ATH files or outputs. Runs on BAR-ATH print only validity: row count, official checker pass, identical twice, sha256. pytest keeps `--tb=line --show-capture=no`.
- **External data only:** every setting, threshold, weight and trained model comes from the 7 IDSSE matches (CC BY 4.0, cite it). Leave one match out for every score: the table or model that scores match m never saw match m.
- **Reproducibility:** the submission must rebuild byte for byte on the organisers' machine. Any trained model is committed to the repo with the script, seed and data that made it (organiser issue #11). Inference on CPU, deterministic.
- **Process:** superpowers. A written plan, test first, Ashen commits each task (Summary and Description, no Co-Authored-By lines), one fresh review on Opus at the end, vault notes and the team page updated at each milestone. No em or en dashes anywhere.
- **Time:** build Thu 5 to Tue 10 Nov. Ashen is mostly away Fri evening to Sun (quiz); long runs go in the background then. Plan 4 starts Wed 11 Nov; code freeze Fri 13 Nov.

## 4. Practice data made fairer first (degrade v3)

Before any tuning, two changes so the practice mess is closer to real tracking. Both are decided before any score is seen, and the baseline and Plan 2 are re-scored on v3 so every comparison is like for like.

1. **Ball wobble.** Real trackers wobble the ball as well as the players; IDSSE's ball is clean, which flatters touch detection and the "ball still" clue. Add the same slow-drift noise to the ball as to players (same time constant, 1 s), sized per level: none 0, light 0.10 m, medium 0.25 m, heavy 0.50 m. *(Claude's call: same sizes as player wobble, chosen before seeing any score.)*
2. **Two hiding styles, both counted.** The plan 2 review found that degrade v2 hides the ball on 84% of possession changes at heavy (60% overall), because changes happen in crowds and in the air. We cannot check how often real trackers do this without looking at our match, which is off limits. So instead of guessing, every candidate is scored under **both** hiding styles: v2's (half on hard frames) and fully random. *(Claude's call.)*

This gives 7 matches x 4 levels x 2 hiding styles x 2 seeds = 112 practice cases per evaluation.

## 5. The steps, in order of payoff per day

**How every number is measured (honest, nested).** Each practice case is degraded with two seeds (0 and 1), under both hiding styles, at all four levels. Learned tables and models are trained without the scored match, as in Plan 2. **Settings are chosen without it too:** every search point is scored on every match, and for each match m the setting that does best on the other six is then scored on m. The mean of those held-out scores is the honest estimate used below. Computing every point's per-match scores once gives this at almost no extra cost. The setting frozen for the submission is the one that does best on all seven. *(Claude's calls, from the builder check: nested selection, two seeds.)*

Before step A, the seed spread is measured once on the Plan 2 version: the difference in the medium-and-heavy mean between seed 0 and seed 1. If it is more than 0.005, the acceptance bar below rises from 0.01 to twice that spread. Decided now, before any Plan 3 score is seen.

**The acceptance rule (applies to every step).** A step's change is kept only if, on degrade v3, with the honest nested estimate:
- the mean spellF1 over medium and heavy (both hiding styles, both seeds) rises by at least 0.01 (or the raised bar) over the current best, and
- no level (none, light, medium, heavy; averaged over hiding styles and seeds) falls by more than 0.04 (the organisers' noise band).

Otherwise it is dropped and the current best stands. Steps run in this order, each building on the last kept version. *(Claude's calls: the 0.01 bar and averaging over hiding styles and seeds.)*

**A. Wider settings search.** Only the ratio of each cost to the temperature matters (plan 2 review), so fix the temperature at 1 and search the switch, to-dead and from-dead costs (to-dead and from-dead separately) on a wider log grid, because the plan 2 winner sat on the grid edge. Also search the clue settings plan 2 left at starting guesses: touch radius and speed jump, control radius and minimum time, how long the last touch is carried, and the ball-out margin. Search method is the builder's choice (grid or coordinate search), as long as every point is scored by the acceptance rule's measure.

**B. Fill short ball gaps.** When the ball is unseen for a short stretch between two seen frames, join the last and next positions with a straight line, for gaps up to G frames (G searched, for example 0.5 to 3 s). Clues then run through short gaps. Interpolated frames are marked so that nothing treats them as a measured touch. Allowed: the rules say improving the tracking with our own algorithm is legitimate.

**C. Easier switches after a touch.** Today a change of team costs the same on every frame. Make the H-to-A switch cheaper on frames near a touch by an A player (and A-to-H near an H touch), so changes land where a touch actually happened. The discount and the window width are searched. Decoded with the same per-half Viterbi.

**D. A smarter frame scorer.** Replace (or blend with) the 54-cell tally's **H versus A part** by a small learned model. Trained only on IDSSE **live frames** (degraded, mirrored home and away), leave one match out for every score, Viterbi still on top.
- **What it learns, and what it does not (Ashen's 3 Nov decision 4 still holds):** it learns only who has the ball while play is live, because that is the honest part of IDSSE. Its output gives the H and A scores; as today, the team part of the D score is the average of the H and A likelihoods, so team clues never push towards or against a dead ball. The dead-ball side stays exactly as now: players' speed learned from all frames, plus the fixed ball hints (ball out, ball still).
- **Clues allowed:** only ones computed the same way from positions on both datasets (the 3 Nov rule), for example nearness margin, last touch and its age, control and its duration, ball speed, pitch zone, time since the ball was last seen, and whether the position is interpolated. **Never ball-out or ball-still**: IDSSE freezes its dead ball, so a model would learn a shortcut that real tracking does not give.
- **Hint weight:** decision 5's written rule keeps its meaning: the median, over IDSSE live frames, of the absolute per-frame H versus A score difference, now of whichever scorer is in use. Still never tuned.
- **Model:** small and regularised, settings fixed in advance, deterministic with a fixed seed. Model family is the builder's choice; it must be explainable in the technical report (for example which clues matter most).
- **Kept only if** it passes the acceptance rule against the best version after step C. If not, the tally stays and the day is the only cost.

**E. Re-tune the final pieces together.** After A to D, one last search of the costs on the final version (settings interact). Freeze the result into `hmm_v2.json` (and any model file).

**F. Review fixes (alongside, not gated).** From the plan 2 review: compute clues per half so nothing crosses the half-time gap; a decoder test with the ball missing most of a half; a pass flying 1 m from an opponent stays with the passer's team; a broken Smart Tagging time does not show its value in the error; a glitch next to no player cannot hide a real touch next to one.

## 6. What comes out

- `hmm_v2.json` (and a committed model file if step D is kept), used by `make_submission --method hmm`, still valid and byte-identical on two runs.
- The scoreboard on degrade v3 for the baseline, Plan 2, and each kept step, both hiding styles, recorded in the vault Scoreboard with a plain verdict per level.
- A short record of every step tried, kept or dropped, with its numbers, for the technical report's "what we tried and dropped" section (SUBMISSION.md asks for it).
- Vault: hub milestone, Plan 3 explained note in plain words, Plain words guide terms; team page republished.

## 7. Decided by Ashen

- Goal is the top 20, so the highest spellF1 we can get (macroF1 as tie-break).
- Sure gains first; nothing that can take days and come back empty.
- Team shape and heatmaps wait for the later rounds.
- Gap filling and the smarter scorer are in Plan 3, smarter scorer last.
- Plan 3 makes it better; Plan 4 ships it (doc A with technical report B behind it, README, rebuild check, going public around 14 to 15 Nov, the form). An optional experiment only if time allows after Plan 4.
- Repo stays private until we submit.

## 8. Claude's calls (marked so Ashen can check them)

- Degrade v3: ball wobble sized like player wobble; both hiding styles counted.
- The acceptance rule's numbers: +0.01 on medium and heavy, -0.04 guard per level.
- Temperature fixed at 1; to-dead and from-dead searched separately.
- Order of steps by payoff per day; step F not gated.
- New dependency for step D only if a model needs it (for example scikit-learn, pinned).
- From the builder check: settings chosen with each scored match left out (nested), two degrade seeds with a seed-spread check that can raise the bar, and the smarter scorer limited to H versus A on live frames with the hint weight rule carried over.

## 9. Left to the builder

Search method and grid points, the gap-length range, the touch window shape, the model family and its clue list within section 5D's limits, code layout, test names, and how the step-by-step record is stored, as long as each choice is measured with the acceptance rule and explained in the technical report.
