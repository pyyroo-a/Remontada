import pytest

from remontada.smart_tagging import Tag, load_smart_tagging, parse_smart_tagging

SAMPLE = """<file><ALL_INSTANCES>
<instance><ID>1</ID><start>447.38</start><end>452.4</end><code>SET PIECES</code>
  <label><group>Team</group><text>N/A</text></label><label><group>Type</group><text>Throw-in</text></label></instance>
<instance><ID>2</ID><start>10.0</start><end>12.02</end><code>Build Up</code>
  <label><group>Team</group><text>barcelona</text></label></instance>
<instance><ID>3</ID><start>20.0</start><end>21.0</end><code>DEFENSIVE TRANSITION</code>
  <label><group>Team</group><text>Atletic Club</text></label></instance>
</ALL_INSTANCES></file>"""


def test_parse_frames_codes_and_teams(tmp_path):
    (tmp_path / "t.xml").write_text(SAMPLE, encoding="utf-8")
    tags = parse_smart_tagging(tmp_path / "t.xml")
    assert [t.code for t in tags] == ["SET PIECES", "BUILD UP", "DEFENSIVE TRANSITION"]
    assert (tags[0].start_frame, tags[0].end_frame, tags[0].team, tags[0].type) == (11184, 11310, -1, "Throw-in")
    assert (tags[1].start_frame, tags[1].end_frame, tags[1].team) == (250, 300, 0)
    assert tags[2].team == 1


def test_bad_time_does_not_show_its_value(tmp_path):
    (tmp_path / "t.xml").write_text(SAMPLE.replace("447.38", "44x.38"), encoding="utf-8")
    with pytest.raises(ValueError) as err:
        parse_smart_tagging(tmp_path / "t.xml")
    assert "44x" not in str(err.value) and "instance 1" in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__


def test_tag_repr_hides_the_match():
    # like FrameTable: a printed tag or a failed test must not show match data
    assert repr(Tag("BUILD UP", 250, 300, 0, None)) == "Tag(...)"


@pytest.mark.realdata
def test_bar_ath_smart_tagging_summary():  # counts published in DATA.md
    tags = load_smart_tagging()
    assert len(tags) == 638 and len({t.code for t in tags}) == 14
    assert sum(t.code == "SET PIECES" for t in tags) == 59
    assert [sum(t.team == k for t in tags) for k in (0, 1, -1)] == [314, 288, 36]
    kickoffs = [t for t in tags if (t.type or "").lower() == "kick off (start)"]
    assert all(any(t.start_frame <= f <= t.end_frame for t in kickoffs) for f in (11185, 113742))
