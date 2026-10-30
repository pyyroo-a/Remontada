import numpy as np
import pytest

from remontada.load_idsse import load_idsse
from remontada.vendor import possession_metrics as pm

LATE_SLOT = 2  # slots follow first appearance in the file: home, away, late home sub

INFO = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n'
    '<PutDataRequest><MatchInformation>\n'
    '<General MatchId="DFL-MAT-TEST01" HomeTeamId="CLU-H" GuestTeamId="CLU-A" />\n'
    '<Environment PitchX="105.00" PitchY="68.00" />\n'
    '</MatchInformation></PutDataRequest>\n'
)


def _frameset(section, team, person, frames):
    lines = ['<FrameSet GameSection="{}" MatchId="DFL-MAT-TEST01" TeamId="{}" PersonId="{}">'
             .format(section, team, person)]
    lines += frames
    lines.append("</FrameSet>")
    return lines


def _player(n, x):
    return '<Frame N="{}" T="t" X="{:.2f}" Y="1.00" D="0.00" S="0.00" A="0.00" M="1"/>'.format(n, x)


def _ball(n, possession, status):
    return ('<Frame N="{}" T="t" X="0.00" Y="0.00" Z="0.10" D="0.00" S="0.00" A="0.00" M="1" '
            'BallPossession="{}" BallStatus="{}"/>'.format(n, possession, status))


def write_synthetic_idsse(root, truncate=False):
    first, second = range(10000, 10004), range(100000, 100004)
    lines = ['<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>', "<PutDataRequest>", "<Positions>"]
    for section, ns in (("firstHalf", first), ("secondHalf", second)):
        lines += _frameset(section, "referee", "REF", [_player(n, 9.0) for n in ns])
        lines += _frameset(section, "CLU-H", "P-H1", [_player(n, -1.0) for n in ns])
        lines += _frameset(section, "CLU-A", "P-A1", [_player(n, 1.0) for n in ns])
        late = ns[2:] if section == "firstHalf" else ns
        lines += _frameset(section, "CLU-H", "P-H2", [_player(n, -2.0) for n in late])
    states = [(1, 1), (1, 1), (2, 0), (2, 1), (2, 1), (2, 1), (1, 1), (1, 0)]
    balls = list(zip(list(first) + list(second), states))
    lines += _frameset("firstHalf", "BALL", "B", [_ball(n, p, s) for n, (p, s) in balls[:4]])
    lines += _frameset("secondHalf", "BALL", "B", [_ball(n, p, s) for n, (p, s) in balls[4:]])
    lines += ["</Positions>", "</PutDataRequest>"]
    if truncate:
        lines = lines[: len(lines) // 2]
    (root / "DFL_02_01_matchinformation_DFL-COM-000009_DFL-MAT-TEST01.xml").write_text(INFO, encoding="utf-8")
    (root / "DFL_04_03_positions_raw_observed_DFL-COM-000009_DFL-MAT-TEST01.xml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_synthetic_idsse(tmp_path):
    table, labels = load_idsse("TEST01", write_synthetic_idsse(tmp_path))
    assert table.n_frames == 4 + 1 + 4
    assert labels.tolist() == [pm.H, pm.H, pm.D, pm.A, pm.X, pm.A, pm.A, pm.H, pm.D]
    assert table.graded_runs == [(0, 4), (5, 9)]
    assert table.player_team.tolist() == [0, 1, 0]
    assert np.isnan(table.player_xy[0:2, LATE_SLOT]).all()
    assert not np.isnan(table.player_xy[2, LATE_SLOT]).any()
    assert np.isnan(table.ball_xy[4]).all()


def test_truncated_file_raises_clearly(tmp_path):
    with pytest.raises(ValueError, match="positions"):
        load_idsse("TEST01", write_synthetic_idsse(tmp_path, truncate=True))


@pytest.mark.realdata
def test_real_idsse_koeln_bayern():  # values measured 29 Oct 2026
    table, labels = load_idsse("J03WMX")
    assert table.n_frames == 70708 + 1 + 75259
    counts = {c: int((labels == c).sum()) for c in (pm.H, pm.A, pm.D)}
    assert counts == {pm.H: 39111, pm.A: 46258, pm.D: 60598}
