# Plan 4: ship the Remontada submission (design)

Date: 12 Nov 2026. Approved by: Ashen, 12 Nov 2026. Team Remontada, More than a Hack 2027, Phase 1 (Initial Challenge).

## 1. Intent

Plans 1 to 3 built and tuned the possession decoder; `main` now ships `src/remontada/config/hmm_v2.json` (merged 6 Nov, pull request #3). Plan 4 turns that into a complete, valid submission and submits it **early (14 Nov)**, well before the 17 Nov deadline.

What Ashen wants a reader (an organiser now, a judge in the later rounds) to come away with: **we did careful, honest work, and the output is strong and meets every expectation the organisers wrote down.** The score alone decides the Phase 1 ranking, but the organisers say the repo and the document are what make the score mean something, so they get the same care as the code.

## 2. What the organisers require (read 12 Nov from upstream `SUBMISSION.md`, `RULES.md`, `FAQ.md` at commit f8f2629, unchanged since 23 Oct)

1. **The possession file** `remontada.csv`: header `frame,label`, exactly 199,319 rows, frames 0 to 199318 in order, labels H/A/D/X, no blanks. Validate the exact file we upload with the official `validate_submission.py`.
2. **A public code repository**: source, pinned dependencies, installation and execution instructions, and the command that regenerates the file. Their bar: someone with the data bundle and our repository reproduces our file, "not approximately", the same file. No match data, and no derived data that would rebuild it (frame-level tracking exports, clipped video). Intermediate files are allowed.
3. **A technical document**, "a few pages is plenty", PDF or Markdown, named `remontada.pdf`, covering: the approach and why; which assets we used and how (including the ones tried and dropped); training data (every labelled dataset, its source, who labelled it, how much; none from the provided game); the possession rules we implemented (loose balls, challenges, restarts); limitations; reproduction.
4. **The form** (link by email): team name exactly as registered ("Remontada"), the CSV, the public repo URL, the document. One response per team, editable until the deadline. Needs a Google account.

Also: the 4 Nov rules clarification email overrides every earlier document (see the vault note Organiser rulings). Issues are now disabled on the upstream repo, so any new question goes by email.

## 3. Goals

- **G1. The document** `docs/remontada.md` (source of truth, in the repo) exported to `remontada.pdf` for the form.
  - One document, a few pages (Ashen chose one document over a two-layer brief plus report).
  - Opens with **"Remontada in one minute"** (about half a page, plain words): what the system does, how we kept it honest, the practice numbers, and a plain statement that practice numbers on other matches are not a prediction of our score on the Barça match.
  - Then the organisers' **six headings, in their order and with their names**, so a checker can tick them off.
  - **Authors:** "Team Remontada" with a placeholder list of members that Ashen fills in later. Ashen must be told exactly how: edit the authors block in `docs/remontada.md`, then rerun the export command.
  - Deep detail is linked, not pasted: see G2.
- **G2. A results page** in the repo (for example `docs/results.md`): the full practice tables (degrade v3 and v4, by level, by hiding style and by ball style), every Plan 3 step kept or dropped with its numbers, the selection gap, and how each number can be rerun. The document links to it.
- **G3. The README** a stranger can follow (checked on Windows 11; macOS and Linux steps marked as not checked, see section 7): Python version, install from the pinned dependencies, where to put the data bundle (`MTAH_DATA_DIR`), the one command that writes `remontada.csv`, the expected sha256, how to run the official validator, how to run the tests, and the IDSSE citation (CC BY 4.0).
- **G4. The rebuild check:** a fresh clone in a new folder outside the repo, a fresh virtual environment from the pinned dependencies, the README's command against the data bundle, then the same sha256 as the reference file and a pass from the official validator. Recorded in the ledger and the vault.
- **G5. The pre-1-Nov note:** add one line to `docs/superpowers/specs/2026-10-30-possession-pipeline-design.md` at "What the baseline taught us" saying those numbers describe the official baseline on the public 5 minutes, were written on 24 to 30 Oct for understanding only (as the FAQ then allowed), predate the 4 Nov clarification, and were never used to tune or choose anything. Ashen decided: keep the record, add the note.
- **G6. Going public and submitting (14 Nov):** Ashen flips the repo to public (a settings change, Ashen's action). Claude then checks the public repo page from outside (no data files visible, README renders). Ashen submits the form with the exact validated file.
- **G7. The vault and the team page** updated as usual (hub milestones and decisions, a "Plan 4 explained" note, Plain words for any new term, the pinned team page republished).

## 4. Non-goals

- No new modelling, tuning or experiments. `hmm_v2.json` ships unchanged. The smooth versus rough ball question (see the vault note Open questions) goes into the document's limitations as an honest bracket, not into a late change.
- No two-layer brief. A plain-language pitch is for the later rounds if we reach the top 20.
- No history rewrite: Ashen decided to keep the commit identities as they are (33 commits carry a university name and email; accepted).
- No fixes to the deferred Plan 3 minors (loader strictness, mirror bin edge); they are mentioned only where relevant.

## 5. Anti-goals (would make Plan 4 a failure even if everything "works")

- Any number in the document, README or results page that we did not measure, or that was measured on our match. The only facts about our match that may appear: the file is valid (199,319 rows, official validator passes), identical on every run, and its sha256.
- Overselling: implying the practice scores predict our ranking, or hiding a limitation.
- Any match data, or anything frame-level built from it, reaching git or the public repo.
- A README command that does not reproduce the exact file on a fresh machine.
- Em dashes or en dashes anywhere (team style rule).

## 6. Constraints

- **Data rules (unchanged):** Claude never opens, prints or reads BAR-ATH files under `data\tactical` or `data\broadcast`, `data\README.md`, or anything in `outputs\`. Runs on BAR-ATH print only "rows N valid" and the sha256. No match data or extracts to any cloud AI.
- **Commits:** Ashen commits and pushes; Claude gives the files to tick, a Summary and a Description after each task and waits for "committed". No Co-Authored-By lines.
- **Branch:** work on a new branch `plan-4-ship` cut from `main` at 5ffb500 (Claude's call: same flow as Plans 1 to 3).
- **Time:** build 12 to 13 Nov; go public and submit 14 Nov; 15 to 17 Nov is buffer (the form stays editable).
- **Style:** plain, friendly words; no em or en dashes; every number checked against a file or a run.

## 7. Decided by Ashen

- One document, with a one-minute summary on top and the organisers' six headings; deep detail linked from the repo.
- Markdown source in the repo, exported to `remontada.pdf` for the form.
- Submit early (14 Nov).
- Keep commit identities as they are; add the note to the 30 Oct design doc.
- Authors: the whole team, with names filled in later by Ashen.
- An AI coding assistant is fine to use (10 Nov); the document may say so plainly, together with the rule that no match data went to any cloud AI.
- **Licence: MIT** (12 Nov). A `LICENSE` file at the repo root (copyright "Team Remontada", the year 2026) and the licence field in `pyproject.toml`. The vendored scorer keeps its own upstream MIT notice, and IDSSE-derived files (the tally, `results/`) keep the CC BY 4.0 attribution in the README. Ashen gives the teammates a heads-up.
- **Reproducibility, option (a)** (12 Nov): pin the full dependency tree (every package the environment installs, with exact versions, not just numpy, pandas and pytest), run the fresh-clone check (G4) on Ashen's Windows 11 PC, and have the README and the document say plainly that the sha256 was checked on Windows 11 with Python 3.11. No cross-platform claim beyond what was checked; macOS and Linux setup steps can be listed as "should work, not checked".

## 7b. Settled after the builder check (12 Nov)

- **Numbers about our match in the old docs (answered by checking).** The 30 Oct design's 40.7% ball missing (59.3% seen), the Smart Tagging counts (638 instances; Barcelona 314, Atletic Club 288, N/A 36; set pieces 26, 12, 10, 5, 4 and 2) are the organisers' own published figures in upstream `DATA.md` at f8f2629. They stay; the G5 note says so. The only numbers there that we measured ourselves on our match are the "What the baseline taught us" lines (the official baseline on the public 5 minutes, 24 Oct), and the G5 note covers that whole block.
- **The vendored public 5 minutes (Claude's call):** `src/remontada/vendor/BAR-ATH_possession_gt_public.csv` stays. It is the organisers' own file, published in their repo and covered by their MIT licence ("the public 5-minute ground truth in evaluation/gt/"), it is already in our history (removing it from the tree would not remove it from git), and it is never used to score our output. The README's vendor section already says this; the document repeats it in one line.
- **Where numbers come from (Claude's call):** every number in the document and the results page traces to a committed file in `results/`, a committed script that prints it, or the organisers' docs. The Plan 3 ledger is not in the repo (a copy sits in the session notes), so it is not cited as a source. Plan 2's degrade v2 numbers and the Plan 2 clue statistics are history and are left out; Plan 2 appears through its rows in the degrade v3 and v4 tables in `results/`.
- **`make_submission` defaults to the baseline method**, so the README command must pass `--method hmm` (or the default changes; builder's call, with a test).

## 8. Left to the builder

- The exact wording and structure inside each of the six headings, and the length (aim: about 4 to 6 pages as PDF).
- **The possession rules section** must describe what our code actually does, read from the code and its tests (touch and control clues, carry, the switching cost that keeps a loose ball with the last team, dead-ball clues and the ball-out margin, how a restart is treated, the D state costs), plus which labelling convention we tuned towards (IDSSE `BallPossession` and `BallStatus`) and how that relates to the organisers' definition in `CHALLENGE.md`. If the code does not draw a line somewhere (for example on challenges), say so.
- How the PDF is made (pandoc 3 is installed; it may need a PDF engine, or go through HTML and a headless browser). The export must be a single documented command.
- Whether the results page is one file or a few.
- How to run the official validator (upstream `evaluation/validate_submission.py` from the data bundle or the upstream repo; it is not vendored today).
- Small README fixes found along the way (setup notes per platform were an open item from Plan 1).

## 9. Done looks like

- `docs/remontada.md` and `remontada.pdf` cover the six headings, open with the one-minute summary, carry the authors placeholder, and every number traces to a file in `results/` or a ledger line.
- README, results page and the 30 Oct note in place; IDSSE cited.
- The rebuild check passes: same sha256, official validator passes.
- One fresh Opus review of the whole branch, findings fixed.
- Ashen merges `plan-4-ship`, flips the repo to public, Claude checks the public page, Ashen submits the form on 14 Nov.
- Vault and team page updated.
