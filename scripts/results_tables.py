"""Print every practice table on docs/results.md from the committed results/ files.

    python scripts/results_tables.py

All numbers are spellF1 on the 7 IDSSE matches (CC BY 4.0) made messy on purpose, every one nested
(each match scored with a tally learned without it and settings picked on the other six). Nothing
here was measured on our match. tests/test_results_tables.py checks that docs/results.md shows
exactly these tables.
"""
from pathlib import Path

from remontada import practice
from remontada.pipeline import settings_from_dict

ROOT = Path(__file__).resolve().parents[1]
STEP_NAMES = {
    "A-clues": "A. Wider search of costs, then clues",
    "B-gaps": "B. Fill short ball gaps",
    "C-touch": "C. Cheaper switches next to a touch",
    "D-scorer": "D. A small learned scorer",
    "E-final": "E. Costs once more on the final version",
}


def _load_state(folder):
    import json
    return json.loads((Path(folder) / "state.json").read_text(encoding="utf-8"))["steps"]


def _mh(levels):
    return sum(levels[lv] for lv in practice.MEDIUM_HEAVY) / len(practice.MEDIUM_HEAVY)


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _level_table(columns, bold):
    """columns: [(title, {level: value})]; bold: index of the column that ships."""
    def cell(i, value):
        return "**{:.4f}**".format(value) if i == bold else "{:.4f}".format(value)
    rows = [[lv] + ["{:.4f}".format(levels[lv]) for _, levels in columns] for lv in practice.LEVEL_NAMES]
    rows.append(["**medium + heavy**"] + [cell(i, _mh(levels)) for i, (_, levels) in enumerate(columns)])
    return _table(["Messiness"] + [title for title, _ in columns], rows)


def _a_clues_names(folder):
    contexts = _load_state(folder)["A-clues"]["contexts"]
    return {c: practice.point_name(settings_from_dict(s)) for c, s in contexts.items()}


def v3_levels(folder):
    """Degrade v3 (rough ball, 112 cases): baseline, Plan 2, and Plan 3 step A rescored with the
    decoder's 6-decimal rounding (scored inside search-final.json)."""
    folder = Path(folder)
    start = practice.load_results(folder / "start.json")
    final = practice.load_results(folder / "search-final.json")
    plan3, _ = practice.nested_step(final, {c: [n] for c, n in _a_clues_names(folder).items()})
    return _level_table([("Baseline", practice.level_means(start["baseline"])),
                         ("Plan 2", practice.level_means(start["plan2"])),
                         ("Plan 3 (hmm_v2.json)", plan3)], bold=2)


def v4_levels(folder):
    """Degrade v4 (rough and smooth ball, 224 cases): baseline, Plan 2, the shipped hmm_v2.json, and
    the step A rerun (nested)."""
    folder = Path(folder)
    start = practice.load_results(folder / "start.json")
    rerun = _load_state(folder)["A-clues"]["levels"]
    return _level_table([("Baseline", practice.level_means(start["baseline"])),
                         ("Plan 2", practice.level_means(start["plan2"])),
                         ("hmm_v2.json (ships)", practice.level_means(start["v2"])),
                         ("Step A rerun", rerun)], bold=2)


def v4_by_style(folder):
    """Medium + heavy on degrade v4, split by ball style and by hiding style."""
    start = practice.load_results(Path(folder) / "start.json")
    labels = {"rough": "rough ball", "smooth": "smooth ball",
              "hard": "ball hidden where trackers lose it", "random": "ball hidden at random"}
    rows = []
    for part in (1, 0):  # ball first, then hiding
        for value in dict.fromkeys(practice.split_style(st)[part] for st in practice.STYLES):
            row = [labels[value]]
            for point in ("plan2", "v2"):
                only = {c: v for c, v in start[point].items() if practice.split_style(c[2])[part] == value}
                row.append("{:.4f}".format(_mh(practice.level_means(only))))
            rows.append(row)
    return _table(["Style (medium + heavy)", "Plan 2", "hmm_v2.json"], rows)


def steps(folder):
    """Every Plan 3 step on degrade v3, nested medium + heavy and the verdict."""
    state = _load_state(folder)
    rows = [[STEP_NAMES[step], "{:.4f}".format(_mh(state[step]["levels"])), state[step]["verdict"]]
            for step in STEP_NAMES]
    return _table(["Plan 3 step", "Medium + heavy (nested)", "Verdict"], rows)


def selection_gap(plan3, plan3b):
    """The shipped choice scored on the matches it was picked on, against the nested estimate."""
    rows = []
    for title, folder, results_file in (("Plan 3 step A (degrade v3)", plan3, "search-final.json"),
                                        ("Step A rerun (degrade v4)", plan3b, "search-clues.json")):
        results = practice.load_results(Path(folder) / results_file)
        in_sample, nested = practice.selection_gap(results, _a_clues_names(folder))
        rows.append([title, "{:.4f}".format(in_sample), "{:.4f}".format(nested),
                     "{:.4f}".format(in_sample - nested)])
    return _table(["Search", "In-sample", "Nested", "Gap"], rows)


def tables(root):
    root = Path(root)
    plan3, plan3b = root / "plan3", root / "plan3b"
    return {"v3-levels": v3_levels(plan3), "v4-levels": v4_levels(plan3b),
            "v4-by-style": v4_by_style(plan3b), "steps": steps(plan3),
            "selection-gap": selection_gap(plan3, plan3b)}


def main():
    for name, table in tables(ROOT / "results").items():
        print("<!-- {} -->".format(name))
        print(table)
        print()


if __name__ == "__main__":
    main()
