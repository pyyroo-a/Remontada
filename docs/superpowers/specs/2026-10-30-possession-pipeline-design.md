# Remontada possession pipeline: design

Date: 30 Oct 2026. Status: **approved by Ashen, 30 Oct 2026**. Reviewed once by a fresh builder; its questions are answered in sections 3, 5.2, 5.3 and 10.
Team: Remontada. Challenge: More than a Hack 2027, Phase 1 (Initial Challenge).

Facts quoted here were measured on our data or taken from the official docs and organiser answers; the evidence lives in the team's Obsidian notes (Match data facts, Baseline run 24 Oct, Organiser rulings, IDSSE dataset). Anything not yet checked is marked **unchecked**. Choices marked **(Claude's call)** were made by Claude, not agreed by the team, and are open to change.

## 1. Intent and why

Label every frame of FC Barcelona vs Athletic Club (199,319 frames of the tactical video) as `H` (Barça ball, alive), `A` (Athletic ball, alive), `D` (dead ball) or `X` (no answer), well enough to finish in the **top 20** and reach the Innovation Challenge.

Phase 1 is ranked only on **spellF1** (whole possessions recovered, matched by time overlap at IoU 0.3 to 0.7), with macroF1 as the tie-break. There is no jury, so the score is everything for ranking. But the team's own rule is that **every part must be explainable in plain words**: we must be able to point at any change of possession and say why the system called it. That matters for the technical document, for trusting our own results, and for the later rounds, where a jury does judge.

What the baseline taught us (public 5 minutes, our run of 24 Oct, understanding only):
- With the ball visible and in play, the nearest player is the wrong team on 22% of frames; one 42.4 s Barça possession flips 24 times in that raw signal.
- Dead ball is the biggest single miss: 314 of 1,469 dead frames right; two whole stoppages were called live because the ball stayed visible.
- 80% of its wrong frames had the ball visible; it invents possessions (26 against 17).
- The public window has the ball on 88% of frames, the full graded match on only 59.3%, so the full match is harder.

## 2. Goals, non-goals, anti-goals

**Goals**
1. A submission file that beats the published floor (spellF1 0.2433) by a clear margin on the full match. We cannot measure the full match ourselves, so "clear margin" is judged on our own scoreboard (section 6), with the organisers' estimated noise band of about 0.04 in mind.
2. Byte-for-byte reproducible from the public repo with one command.
3. An explanation for every predicted change of possession and every stoppage (section 5.6).
4. A short technical document that covers everything SUBMISSION.md asks for.
5. Submitted by **16 Nov 2026**, one day before the 17 Nov deadline **(Claude's call)**.

**Non-goals (for this phase)**
- Rebuilding detection, tracking or camera calibration.
- Our own ball detector from video (idea 6). Only as a separately approved spike, trained on other footage, if time allows.
- Using the broadcast recording (not synchronised, 65.8% usable homography) **(Claude's call)**.
- Any user interface or demo. Phase 2 work.

**Anti-goals (failure even if the score looks good)**
- Any setting, threshold or model chosen by looking at scores on the public 5 minutes. They are used only to check the pipeline is correct (frames aligned, labels mean what we think).
- Any hand label on BAR-ATH, for any purpose, including choosing settings (issue #10).
- Any model trained on labels derived from BAR-ATH itself, including its own tracking (issue #2), unless the organisers approve it first in an issue.
- Rules or constants tied to specific frames of BAR-ATH (issue #14).
- Match data, or anything that could rebuild it (frame-level exports, clips), in the repo.
- Match data, or extracts of it, shown to or uploaded to any hosted LLM, including the assistant building this.
- A part we cannot explain, or a number we cannot reproduce.

## 3. Constraints

- **Rules:** RULES.md and the organiser answers on issues #2, #8, #10, #11, #14, #15, #16. Future frames may be used (#15).
- **No match data to cloud LLMs:** OVERVIEW.md and the FAQ, quoting the Terms and Conditions, say participants "shall not upload, disclose or otherwise make available to any third-party hosted or cloud-based LLM any videos, datasets, confidential information or other restricted materials provided in connection with the Competition, or any extracts thereof". Claude, which builds this, is a cloud-hosted LLM. So Claude only ever sees code, the public repo's own files (docs, public ground truth, shipped baselines), IDSSE and summary results (scores, counts, error messages). No script run while building prints raw rows, positions, frames, tags or video from the BAR-ATH bundle, and nobody pastes such data into any hosted LLM. An LLM used inside the pipeline on BAR-ATH data must run locally.
- **Reproducibility (#11):** the CSV must regenerate byte for byte. Commit any trained weights, run inference on the CPU, pin every dependency, fix every seed. Training itself need not be exactly reproducible but its code must be published. No runtime limit.
- **Data location:** match data stays in `D:\hackathon\more than a hack\data` (outside the repo); the code finds it through `MTAH_DATA_DIR`, as the official starter does. IDSSE is downloaded by a script into a folder outside git as well.
- **IDSSE (#16):** allowed for training and tuning; must be cited (CC BY 4.0) in the README and technical document; every setting chosen on IDSSE or other games.
- **Language:** Python 3.11, numpy and pandas, plus only what a part truly needs **(Claude's call)**. The official scorer runs on numpy and pandas only.
- **Scorer:** the official `evaluation/possession_metrics.py` is the definition of every metric. We vendor a copy pinned to upstream commit `f8f2629`, with its MIT licence notice, rather than reimplementing it **(Claude's call)**.
- **Dates:** registration closes 10 Nov 2026; submission deadline 17 Nov 2026.
- **Who builds (decided by Ashen, 30 Oct):** one builder sequence. Claude builds each step; Ashen reviews and commits it. The frame table and the scoreboard come first, behind clear interfaces, so teammates can later take separate parts (for example dead-ball cues or the team-shape cue) without touching the same files.
- **Commits:** Claude never commits or pushes; it hands over the files to tick, a Summary and a Description at each milestone. Ashen commits. How the rest of the team commits is still open.

## 4. The approach in one picture

```
data (tactical ATD, Smart Tagging)      IDSSE (7 Bundesliga matches)
            |                                     |
            v                                     v
       frame table  <------ same shape ------  frame table  --> degrade to look like ours
            |                                     |
            v                                     v
     evidence (cues)                         evidence (cues)
            |                                     |
            v                                     v
     decoder (HMM + Viterbi)  <-- settings --  tune and score against IDSSE labels
            |
            v
   remontada.csv  +  explanations  +  validate_submission.py
```

The ideas from the brainstorm map onto it like this: idea 1 (touches, not distance) and idea 3 (dead balls) are evidence; idea 2 (long, clean possessions) is the decoder; idea 5 (honest testing on other matches) is the scoreboard; idea 4 (team shape) is an extra evidence source added after the core works.

## 5. The parts

### 5.1 Frame table
One in-memory table per match, the same shape for BAR-ATH and IDSSE: per frame, the ball position (x, y, and z where known; missing as NaN), and every tracked object's position, speed and team. Coordinates in metres with a fixed orientation **(Claude's call)**. BAR-ATH tracking is 0 to 1 (pitch 105.3 x 68.0 m in `games.py`); IDSSE is already in metres with the origin at the centre spot (pitch 105 x 68). Frame alignment for BAR-ATH follows DATA.md: video frame = ATD `frame_id - 1`; Smart Tagging seconds x 25, rounded. Players are anonymous track slots on BAR-ATH, so nothing may depend on player identity.

### 5.2 IDSSE labels and degradation
- **Labels:** each IDSSE ball frame has `BallStatus` (1 alive, 0 dead) and `BallPossession` (1 home, 2 away; checked against one match's pass events, 91% and 94% agreement). Our label per frame: `D` if dead, else `H` or `A`.
- **Which label we tune on (Claude's call):** IDSSE's own `BallStatus` and `BallPossession`, as they are. Reason: CHALLENGE.md says our ground truth is made mechanically from corrected tracking, `D` where it marks the ball dead and otherwise "the team the tracking names as owning it". IDSSE's two fields are the same kind of thing, a tracking provider's per-frame dead flag and ball owner, so they are the closest honest stand-in. The public 5 minutes are not used to pick or adjust the label convention.
- **Robustness check, not a target:** in the one match checked, IDSSE possessions were shorter and more frequent (25 in the first 5 minutes against our public truth's 17; one match, may not generalise). We report possession counts and lengths across all 7 matches, and we also score our pipeline against a stricter label rebuilt from IDSSE events (possession changes only on a completed action by the other team). If our version ranking flips between the two labels, that goes in the technical document as a limitation. Tuning stays on the first label.
- **Won headers (issue #17):** no special rule. Whatever the owner field does with a won header is what we learn, which matches how our ground truth is made. Revisit if the organisers answer #17.
- **Degradation:** IDSSE has the ball on every frame; BAR-ATH misses it on 40.7% of graded frames, and its player tracks fragment and swap. Before scoring on IDSSE, make its tracking look like ours: hide the ball in stretches, add position noise, split and relabel player tracks into anonymous slots. **How the amounts are set (decided by Ashen, 30 Oct):** a range swept on IDSSE, without using anything from BAR-ATH. Each IDSSE match is degraded at several levels, from light to heavy, and a setting is kept only if it holds up across the whole range. This uses nothing from the competition match, so it carries no rules risk, and it does not depend on guessing exactly how BAR-ATH's tracker fails. **Unchecked:** whether degraded IDSSE ranks our versions the same way the full BAR-ATH truth would.

### 5.3 Evidence (cues)
Small, separately testable functions, each producing a per-frame signal with a plain meaning:
- **Touch and control:** frames where the ball's speed or direction changes sharply next to a player; which team is closest and for how long the ball stays near them.
- **Dead ball:** ball outside the pitch lines; ball nearly still for a restart; players slowing down together; ball missing for a long stretch; Smart Tagging set pieces as restart hints (below).
- **Smart Tagging phases:** see below.
- **Team shape (added after the core):** spread, height and compactness of each team, as a hint for who has the ball when the ball is missing. Trained or tuned on IDSSE only.

**Smart Tagging.** The FAQ says to use it: it is Metrica's automatic output from the same video and tracking, not ground truth and not derived from it, with false positives and negatives. Measured on our file: 638 intervals, 14 codes, each with a `Team` tag (Barcelona 314, Atletic Club 288, N/A 36). The 59 `SET PIECES` carry a `Type` (26 throw-ins, 12 corners, 10 goal kicks, 5 free kicks, 6 kick-offs), and every throw-in is tagged `N/A`, so its side must come from elsewhere. DATA.md warns the intervals are phases, not possessions. So we use it as soft evidence, never copied across as labels: a set piece start is a restart hint, and a phase's team tag is a weak hint about who has the ball inside it (attacking codes such as BUILD UP or PROGRESSION for the tagged team; defending codes for the other team, to be confirmed from the code names on the data).

IDSSE has no Smart Tagging, so its weights cannot be tuned there. **(Claude's call):** the weights are fixed in advance by a stated rule written in the config (for example each Smart Tagging hint counts no more than one tracking cue), never adjusted by looking at BAR-ATH answers, and the rule and its reasoning go in the technical document. The public 5 minutes only confirm the times line up (they do for the two set pieces in that window).

### 5.4 Decoder
A hidden Markov model over the states H, A and D, decoded with Viterbi over each half (the whole half, past and future, is allowed). Each cue feeds the emission scores; a switching cost discourages flicker; transitions D to H and D to A model restarts. Every setting (switching costs, cue weights, thresholds) is chosen on degraded IDSSE matches only and then frozen in a committed config file. BAR-ATH runs with that file unchanged.

**Abstentions:** the decoder may output `X` inside a stretch it cannot resolve. On spellF1 an `X` repeats the previous label, which is free inside a possession and costly across a handover, so `X` is used only where the decoder's own uncertainty is high inside one state, never across a change. Whether to use `X` at all is decided on IDSSE.

### 5.5 Output
`remontada.csv`: header `frame,label`, 199,319 rows, frames 0 to 199,318. `X` on every frame outside the two graded halves (video frames 11,185 to 90,651 and 110,808 to 190,197), since those are never scored. Checked with the official `validate_submission.py` before every submission.

### 5.6 Explanations
Alongside the CSV, a small human-readable report: for every predicted change of possession and every stoppage, the frame, the change (for example `H -> A`), and the cues that drove it (for example "touch by an Athletic player at frame 23,410, ball within 1.2 m for 0.8 s"). It contains event times and reasons only, no tracking or positions, so it is safe to commit (issue #14 allows committing small intermediate outputs). This is how we keep the system explainable **(Claude's call on the format)**.

### 5.7 Scoreboard
- **Main:** the official metrics, run on each degraded IDSSE match with our IDSSE labels, reported per match and averaged, with bootstrap intervals from the official code. Settings are chosen by leaving one IDSSE match out at a time, so every reported score is on a match the settings never saw **(Claude's call)**.
- **Sanity only:** the public 5 minutes of BAR-ATH, to confirm frames align and labels mean what we think. Never to choose between versions.
- **Reference:** the official baseline scored the same way, so every version is compared with the floor on the same footing.

## 6. Reproducibility and the repo

- One command regenerates the CSV from the data bundle, for example `python -m remontada.make_submission` **(Claude's call on the name)**. A test runs it twice and checks the two outputs are identical.
- `requirements.txt` with exact versions; Python 3.11; CPU only for inference; seeds fixed wherever randomness appears.
- A script downloads IDSSE (about 3 GB) outside git.
- `.gitignore` keeps any data folder, downloads and large outputs out of git.
- The repo is private while we build and made public before submitting (the form needs a public URL).
- Licence: decided by the team before going public. The vendored scorer keeps its MIT notice.
- The technical document (`remontada.pdf` or `.md`) is written as we go, in `docs/`, and covers every section SUBMISSION.md lists, including the IDSSE citation.

## 7. Building and testing

Test-first, as the team's superpowers workflow requires. Every cue and the decoder are tested on small synthetic frame tables where the right answer is known (for example a pass between teammates must not change possession). Integration tests use short IDSSE slices. Nothing is called done without a measured result on the scoreboard.

## 8. Milestones (Claude's call, to agree with the team)

| By | Milestone |
|---|---|
| 2 Nov | Repo skeleton; BAR-ATH and IDSSE frame tables; scoreboard working; official baseline scored on the public window (must match the README) and on degraded IDSSE |
| 6 Nov | Dead-ball and touch cues plus the HMM decoder (version 1), scored on IDSSE against the baseline |
| 10 Nov | IDSSE convention check across all 7 matches; settings tuned with leave-one-match-out; team-shape cue if time allows. Registration closes. |
| 14 Nov | Frozen config, reproducibility test passing, technical document drafted, repo ready to go public |
| 16 Nov | Submit (`remontada.csv`, repo URL, `remontada.pdf`); 17 Nov is the buffer |

## 9. Left to the builder

The exact cue definitions and thresholds, the form of the emission scores, how degradation is parameterised, code layout inside `src/`, test names, and the format details of the explanation report, as long as each choice is measured on IDSSE and explained in the technical document.

## 10. Still open

- Issue #17: whether a won header without control counts as a short possession in the full-match truth. Unanswered on 29 Oct; our default is in section 5.2.
- Whether IDSSE's possession convention matches ours (section 5.2, robustness check).
- How teammates other than Ashen commit, once they join.
