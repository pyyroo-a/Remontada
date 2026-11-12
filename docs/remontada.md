# Remontada: who has the ball, frame by frame

**Team Remontada**, More than a Hack 2027, Phase 1 (Initial Challenge). 14 November 2026.

**Authors:** Team Remontada: Vimukthi Ashen, Mohammed Sherin, Hussain Abdul Kadhir, Mishal Ashraf, Aman Mohammed

**Code:** https://github.com/pyyroo-a/Remontada (MIT licence). Full practice tables: [docs/results.md](results.md).

## Remontada in one minute

For every frame of FC Barcelona vs Athletic Club we read the provided automatic tracking and decide who has the ball: home (`H`), away (`A`) or dead ball (`D`). A few clues (who touched the ball last, who is controlling it, who is nearest, whether the ball is out or still and the players are slowing down) give each frame a score for each label. A state model then picks the label sequence that fits the clues best while charging a cost for every change of label, so it prefers long, clean possessions over flicker. That matters because the ranking metric counts whole possessions, not frames.

**How we kept it honest.** The rules allow tuning only on outside data, so every setting was chosen on 7 public Bundesliga matches (IDSSE, CC BY 4.0). Their tracking is clean, so we made it messy on purpose at four levels, hiding the ball and wobbling positions the way automatic tracking does, and judged every change on the medium and heavy levels. Every score is **nested**: each match is scored with settings picked on the other six. A change was kept only if it raised that score by at least 0.01.

Practice spellF1, mean of the medium and heavy levels:

| Practice set | Official baseline | Our first decoder (Plan 2) | **What we submit** |
|---|---|---|---|
| Degrade v3: ball wobbles like the players (112 cases) | 0.2156 | 0.3468 | **0.3735** |
| Degrade v4: also a smoothed ball (224 cases); a rerun of our search scored 0.3926 | 0.2170 | 0.3810 | **0.3924** |

These are practice scores on other matches. They are **not a prediction of our score** on the Barça match, which we never scored ourselves: the rules forbid checking against its answers.

## Approach

**Clues from the tracking** (all worked out by our own code from positions, on each half separately so nothing crosses half-time):

- **Touch:** the ball's velocity changes by at least 4.5 m/s (measured over 2 frames each side) while a player is within 1.5 m. The touch goes to the nearest player's team and is carried forward for up to 3 s.
- **Control:** a team's nearest player stays within 2 m of the ball, and nearer than the other team's, for at least 5 frames (0.2 s).
- **Nearness:** how much nearer the closest home player is than the closest away player, in five bins, plus "ball not seen".
- **Dead-ball signs:** the ball more than 0.5 m over a touchline or goal line; the ball still (under 0.5 m/s) for at least 1 s; the players' median speed (they slow down together at a stoppage).

**The tally.** On IDSSE live play we counted how often each combination of touch, control and nearness (54 cells) goes with the home or the away team in possession. Every frame is counted twice, as it is and with home and away swapped, so the tally cannot favour either side. The players' speed part is counted on all frames, live and dead. The ball-out and ball-still signs are not learned: IDSSE freezes the ball exactly when play stops, which real automatic tracking does not do, so they get a fixed weight from a written rule (the typical pull of one frame's team clues).

**The decision model.** Each frame gets a score for `H`, `A` and `D`; a Viterbi decoder finds the sequence with the best total score after paying 20 for every switch between `H` and `A`, 20 to go into `D` and 80 to come out of it. It decodes each half as a whole, using later frames as well as earlier ones, which the organisers allow for this post-match task. Outside the two graded halves we write `X`; inside them we always answer.

**Why this and not something else.** No labels on our match may be used for training or tuning, so anything we learn has to carry over from other matches. A small, explainable model with few settings carries over better than a large one, and every possession change it reports can be traced to the clues that caused it. Possessions are intervals, so a model that charges for switching fits the ranking metric directly.

**How the settings were chosen.** Plan 2 built the decoder and set it roughly. Plan 3 searched the costs and the clue settings properly on grids written down before any score was seen, with the nested scoring above and two random seeds, and tried four more ideas. Only the wider search passed the 0.01 rule; the selection gap (how much better the shipped settings look on the matches they were picked on) is 0.0004, so the search did not fit the practice matches. Details: [docs/results.md](results.md).

## Assets used and dropped

- **Tactical tracking (ATD): used.** It is our only input at prediction time. It is already in pitch coordinates, which we scale to metres (105.3 by 68.0 m), so the homography is not needed.
- **Smart Tagging: built, then switched off.** We wrote a loader and fixed-rule hints from the tag names alone, but overlapping tags stack up and break our rule that a hint never counts for more than one frame's clues, and IDSSE has no tags, so we could not test them on outside data. Not using them was the safer choice.
- **Broadcast video and its tracking: not used.** They run on their own timeline with no mapping to the tactical video.
- **Tactical video: not used** as an input. We did not check our output against it (4 Nov rules clarification).
- **Public 5-minute ground truth:** copied with the official scorer so the copied files can be fingerprinted; never used to score, tune or choose anything.
- **Tried and dropped in Plan 3** (each under the 0.01 bar on messy data): filling short ball gaps with a straight line, cheaper switches right next to a touch, a small learned scorer (logistic regression on 12 position clues) instead of the tally, and a final cost search. A "ball missing for 3 s" dead-ball hint from Plan 2 was dropped too: on messy data it fired more in live play than at stoppages.
- **AI assistance:** we built the code with an AI coding assistant. It saw only our code and summary numbers, never the match data, as the competition terms require.

## Training data

- **IDSSE** (Bassek et al., *Scientific Data* 12, 195, 2025; CC BY 4.0; from [pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data)): 7 complete matches from the 2022/23 Bundesliga 1 and 2, positions of every player and the ball at 25 frames per second, from the DFL and Sportec Solutions (TRACAB optical tracking).
- **Labels used:** the `BallPossession` (home or away) and `BallStatus` (alive or dead) fields that come with that positional data on every frame. We did not label anything ourselves, on any match.
- **How much:** every frame of all 7 matches. For tuning, each match was made messy at 4 levels, with 2 ways of hiding the ball, 2 ball styles and 2 random seeds: 224 practice cases.
- **Nothing from the provided game** was used to train, tune or validate anything. Our match is only the input the decoder reads to write the submission.

## Possession rules

What our system does, in plain words:

- **Gaining the ball.** A team gains the ball through its players' touches, control and nearness, combined in the tally. There is no single rule like "two touches"; a change happens when the clues for the other team are strong and lasting enough to pay the switching cost.
- **Loose balls.** There is no separate "loose" state. While the ball is loose, the last touch is carried forward for up to 3 s and the switching cost keeps the current team, so possession stays with the last team that had it, as the challenge's definition says, until the other team shows clear evidence.
- **Challenges and rebounds.** We have no rule specific to tackles, duels or deflections. A brief contact by the other team usually does not outweigh the switching cost, which is our proxy for "controlled" possession; a sustained one does. Where a duel ends in a clean win, the switch lands where the clues turn.
- **Restarts and dead ball.** `D` is its own state. Ball out by more than 0.5 m, a still ball and slowing players all push towards `D`. Coming out of `D` costs four times as much as going in, so play is called live only when the clues clearly say so, and the team after a restart is decided by the same touch, control and nearness clues. We never try to say who "owns" a dead ball: the answer is always `D`.
- **Ungraded frames** (before kick-off, half-time, after the final whistle) are `X`.

**The convention we tuned towards** is IDSSE's: `BallPossession` and `BallStatus` from its provider's tracking. The challenge says its ground truth also comes from corrected tracking (`D` where the tracking marks the ball dead, otherwise the team it names). That is the same kind of label, but we cannot check how closely the two providers draw the line, because the rules forbid tuning on our match's answers.

## Limitations

- **The ball goes missing exactly where possession changes.** The organisers say the ball is absent on 41% of graded frames, most of all in crowds, aerial duels and tackles. In practice, hiding the ball where trackers lose it costs us far more than hiding the same amount at random (medium + heavy 0.3523 against 0.4324 on degrade v4). No scorer can recover a touch that was never seen.
- **Rough or smooth ball.** Our practice mess cannot know how much the real ball wobbles. With a rough ball our settings beat Plan 2's clearly (0.3745 against 0.3468); with a smooth ball the two are level by our rule (0.4102 against 0.4151). We ship the settings that are best across both (0.3924 against 0.3810).
- **Clean data.** We traded a little clean-data score for messy-data score on purpose (on degrade v4, 0.6491 against Plan 2's 0.6729 with no mess). If our match's tracking were nearly clean, Plan 2's settings would have done slightly better.
- **Our practice mess is a model.** It is our best guess at automatic tracking and is unchecked against our match, by rule. We bracketed the guesses we could not check (two ways of hiding the ball, two ball styles) instead of picking one.
- **Small optimism.** Choosing which steps to keep using the nested scores adds a little optimism to the final practice figure, and each held-out match is one sixth of the tallies used to pick its settings. Both are far below the 0.01 bar.
- **Reproducibility** was checked on Windows 11 with Python 3.11 only.
- **With more time:** read the team shape when the ball is missing (which side is attacking, who is pressing), test Smart Tagging fairly on matches with tags, and detect the ball in the video with a model trained on other footage.

## Reproduction

With the official data bundle and Python 3.11 (full steps in the [README](../README.md)):

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m remontada.make_submission
```

with `MTAH_DATA_DIR` pointing at the data bundle. It writes `outputs/remontada.csv` and prints:

```
rows 199319  valid
sha256 b9c58cf97e4c9c9bf12c298db2760d23dc06717a370cd8e285f87f6ec9e330cd
```

The decoder has no randomness, every package is pinned, and the settings (`src/remontada/config/hmm_v2.json`) and the tally (`evidence_idsse.json`) are committed. The practice numbers can be rerun from the scripts listed on [docs/results.md](results.md).
