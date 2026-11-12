# Practice results

Every number on this page is **spellF1 on 7 public Bundesliga matches** (the IDSSE dataset, CC BY 4.0), never on our match. The rules allow tuning and checking only on outside data, so we make these clean matches messy on purpose and see how each version copes. These are practice scores on other matches, **not a prediction of our score** on FC Barcelona vs Athletic Club.

How to read them:

- **Messiness levels.** `none` is the clean data; `light`, `medium` and `heavy` hide the ball for more and longer stretches (10%, 30% and 60% of frames), wobble the positions and break player tracks. Real automatic tracking loses the ball a lot, so we judge every change on the **medium + heavy** mean.
- **Nested.** Each match is scored with a tally learned without it and with settings picked on the other six matches. So no match helps choose its own settings. (One small, second-order leak: those six were decoded with tallies that included the held-out match, a planned speed shortcut. It is far too small to change a verdict.)
- **Degrade v3 and v4.** Two versions of our practice mess. v3 (112 cases: 7 matches, 4 levels, 2 ways of hiding the ball, 2 random seeds) wobbles the ball like the players. v4 (224 cases) also counts every case with a **smooth ball**, because real trackers usually smooth the ball and we cannot check which is closer to our match.
- **The rule for keeping a change:** the nested medium + heavy mean must rise by at least 0.01, and no level may fall by more than 0.04. Anything smaller is treated as noise.

Every table below is printed by `python scripts/results_tables.py` from the files in `results/`, and a test checks that this page shows exactly what the script prints.

## Degrade v3: Plan 3 against Plan 2 and the baseline

| Messiness | Baseline | Plan 2 | Plan 3 (hmm_v2.json) |
|---|---|---|---|
| none | 0.3081 | 0.6715 | 0.6452 |
| light | 0.2995 | 0.6018 | 0.5845 |
| medium | 0.2638 | 0.4216 | 0.4600 |
| heavy | 0.1675 | 0.2721 | 0.2871 |
| **medium + heavy** | 0.2156 | 0.3468 | **0.3735** |

The official baseline picks the nearest player and smooths. Plan 2 is our first decoder. Plan 3 tuned it properly for messy data: it gives up a little on clean data and gains on medium and heavy, which look most like real tracking. The Plan 3 column is step A scored with the decoder's 6-decimal rounding (scores in `results/plan3/search-final.json`).

## Every Plan 3 step

| Plan 3 step | Medium + heavy (nested) | Verdict |
|---|---|---|
| A. Wider search of costs, then clues | 0.3734 | KEEP |
| B. Fill short ball gaps | 0.3809 | DROP |
| C. Cheaper switches next to a touch | 0.3728 | DROP |
| D. A small learned scorer | 0.3747 | DROP |
| E. Costs once more on the final version | 0.3728 | DROP |

Only step A passed the rule; steps B to E were tried and dropped (B rose by 0.0075, under the 0.01 bar). Step A's 0.3734 here is from before the 6-decimal rounding; rescored, it is 0.3735. Details per step: `results/plan3/state.json`.

## Degrade v4: with a smooth ball too

| Messiness | Baseline | Plan 2 | hmm_v2.json (ships) | Step A rerun |
|---|---|---|---|---|
| none | 0.3081 | 0.6729 | 0.6491 | 0.6493 |
| light | 0.2991 | 0.6067 | 0.5897 | 0.5895 |
| medium | 0.2639 | 0.4618 | 0.4775 | 0.4776 |
| heavy | 0.1701 | 0.3001 | 0.3073 | 0.3076 |
| **medium + heavy** | 0.2170 | 0.3810 | **0.3924** | 0.3926 |

Searching step A again on v4 picked **exactly the settings we ship**, so the rerun is only 0.0002 higher and `hmm_v2.json` stays (`results/plan3b/state.json`).

| Style (medium + heavy) | Plan 2 | hmm_v2.json |
|---|---|---|
| rough ball | 0.3468 | 0.3745 |
| smooth ball | 0.4151 | 0.4102 |
| ball hidden where trackers lose it | 0.3439 | 0.3523 |
| ball hidden at random | 0.4181 | 0.4324 |

Being honest about this table: our lead over Plan 2 comes from the rough ball. With a smooth ball, Plan 2 is 0.0049 ahead, which is under the 0.01 bar, so by our rule the two are level there. For both decoders, hiding the ball where trackers really lose it (in the air, in crowds) is harder than hiding it at random: where the ball goes missing matters more than how much of it does.

## Overfitting check

| Search | In-sample | Nested | Gap |
|---|---|---|---|
| Plan 3 step A (degrade v3) | 0.3739 | 0.3735 | 0.0004 |
| Step A rerun (degrade v4) | 0.3924 | 0.3926 | -0.0003 |

The selection gap is the shipped choice scored on the matches it was picked on, against the nested score. A big gap would mean the search fits the practice matches rather than finding settings that carry over. Both gaps are far smaller than the 0.01 bar.

## Rerunning the numbers

- `python scripts/plan3.py start`, `search-costs`, `search-clues`, `search-gaps`, `search-touch`, `search-scorer`, `search-final` rebuild `results/plan3/` (several hours; needs the IDSSE download, see the README).
- The same commands with `--results-dir results/plan3b`, plus `compare-v2`, rebuild the degrade v4 rerun.
- `python scripts/results_tables.py` prints the tables on this page.
