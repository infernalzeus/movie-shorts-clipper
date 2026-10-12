import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from editorial import text_issues, validate_plan, source_timestamp
from clip_selector import parse_ranges


def valid_plan():
    return {"title": "A book changes Jack's fate", "description": "Scene analysis.",
            "story": "A memory leads to hope.",
            "characters": [{"name": "Jack Harper", "evidence": "Jack Harper", "confidence": "high"}],
            "beats": [{"role": role, "evidence": "You picked it up.",
                       "long_start": s, "short_starts": [s, s + 8],
                       "narration": "Jack's curiosity reveals a trace of humanity.",
                       "byline": "Jack's curiosity changes everything."}
                      for role, s in zip(("setup", "change", "consequence", "payoff"), (100, 130, 160, 190))]}


@pytest.mark.parametrize("bad", ["The narrator notices the target.", "I saw Jack pick it up.", "The primary character feels hope."])
def test_rejects_reproduced_bad_narration(bad):
    plan = valid_plan()
    plan["beats"][0]["narration"] = bad
    with pytest.raises(ValueError, match="placeholders|third-person"):
        validate_plan(plan, 100, 220, 60, "Jack Harper You picked it up.")


@pytest.mark.parametrize("mutation", ["overlap", "outside", "fabricated_evidence", "too_long_byline"])
def test_holds_invalid_edit_plans(mutation):
    plan = valid_plan()
    if mutation == "overlap":
        plan["beats"][1]["short_starts"] = [110, 118]
    elif mutation == "outside":
        plan["beats"][-1]["long_start"] = 219
    elif mutation == "fabricated_evidence":
        plan["characters"][0]["evidence"] = "Jack was in the army."
    else:
        plan["beats"][0]["byline"] = "word " * 13
    with pytest.raises(ValueError):
        validate_plan(plan, 100, 220, 60, "Jack Harper You picked it up.")


def test_equal_planned_runtime_with_different_cut_counts_and_roundtrip_timestamps():
    plan = validate_plan(valid_plan(), 100, 220, 60, "Jack Harper You picked it up.")
    long = [(b["long_start"], b["long_start"] + 15) for b in plan["beats"]]
    short = [(s, s + 7.5) for b in plan["beats"] for s in b["short_starts"]]
    assert len(long) == 4 and len(short) == 8
    assert sum(e - s for s, e in long) == sum(e - s for s, e in short) == 60
    serialized = "; ".join(f"{source_timestamp(s)}-{source_timestamp(e)}" for s, e in short)
    assert parse_ranges(serialized) == short


def test_grounded_third_person_analysis_is_allowed():
    assert not text_issues("Beech describes Jack's curiosity as evidence of remaining humanity.")


def test_comparison_duration_limit_reaches_renderer(monkeypatch, tmp_path):
    import burn
    commands = []
    monkeypatch.setattr(burn, "_run_ffmpeg", lambda cmd: commands.append(cmd))
    burn.burn_subtitles(tmp_path / "raw.mp4", tmp_path / "captions.ass", tmp_path / "final.mp4", output_duration=60)
    assert commands and commands[0][commands[0].index("-t") + 1] == "60"



def test_detailed_context_keeps_plot_character_roles(monkeypatch):
    import metadata
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {"query": {"pages": {"1": {"extract": "== Plot ==\nBeech leads the resistance.\n== Cast ==\nMorgan Freeman as Malcolm Beech.\n== Production ==\nUnneeded production notes."}}}}
    monkeypatch.setattr(metadata, "_SKIP_WEB_CONTEXT", False)
    monkeypatch.setattr(metadata.requests, "get", lambda *a, **k: Response())
    result = metadata.fetch_movie_context("Oblivion (2013 film)", detailed=True)
    assert "Morgan Freeman as Malcolm Beech" in result
    assert "Unneeded production notes" not in result


def test_bounded_planner_request_disables_thinking_and_limits_output(monkeypatch):
    import metadata
    sent = {}
    class Response:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"response": "{}"}
    def post(*args, **kwargs):
        sent.update(kwargs["json"])
        return Response()
    monkeypatch.setattr(metadata.requests, "post", post)
    assert metadata._call_ollama("model", "plan", bounded=True) == "{}"
    assert sent["think"] is False
    assert sent["options"]["num_predict"] == 4096


def test_quote_must_fall_inside_both_selected_versions():
    from editorial import validate_quote_timing
    from subtitles import Cue
    plan = {"beats": [{"evidence": "Jack finds a book", "long_start": 10, "short_starts": [10, 17.5]}]}
    validate_quote_timing(plan, [Cue(12, 14, "Jack finds a book")], 60)
    with pytest.raises(ValueError, match="absent"):
        validate_quote_timing(plan, [Cue(30, 32, "Jack finds a book")], 60)
    with pytest.raises(ValueError, match="WINDOW"):
        validate_quote_timing(plan, [Cue(12, 14, "Jack finds something")], 60)


def test_quote_evidence_ignores_typographic_apostrophes_and_whitespace():
    from editorial import normalized_evidence
    assert normalized_evidence("Jack’s   curiosity") == normalized_evidence("Jack's curiosity")


def test_cut_placement_uses_subtitle_anchors_not_model_arithmetic():
    from editorial import place_quote_ranges, validate_quote_timing
    from subtitles import Cue
    primary = [Cue(t, t+6, "Development number " + str(i)) for i, t in enumerate([105, 140, 175, 210])]
    cues = sorted(primary + [Cue(c.start+10, c.start+13, "Follow up dialogue") for c in primary], key=lambda c: c.start)
    plan = {"beats": [{"evidence": c.text, "long_start": -99, "short_starts": []} for c in primary]}
    place_quote_ranges(plan, cues, 100, 240, 60)
    assert plan["beats"][0]["long_start"] == 104.75
    assert sum(15 for b in plan["beats"]) == 60
    assert sum(sum(b["short_lengths"]) for b in plan["beats"]) == 60
    validate_quote_timing(plan, cues, 60)
    plan["beats"][1]["evidence"] = cues[0].text
    with pytest.raises(ValueError, match="outside its section"):
        place_quote_ranges(plan, cues, 100, 240, 60)


def test_repeated_dialogue_uses_selected_cue_id():
    from editorial import place_quote_ranges, validate_quote_timing
    from subtitles import Cue
    cues = sorted([c for t in [105, 140, 175, 210] for c in [Cue(t, t+6, "Shared line repeats"), Cue(t+10, t+13, "Follow up")]], key=lambda c: c.start)
    plan = {"beats": [{"evidence_id": i} for i in [1, 3, 5, 7]]}
    place_quote_ranges(plan, cues, 100, 240, 60)
    assert plan["beats"][1]["long_start"] == 139.75
    validate_quote_timing(plan, cues, 60)


@pytest.mark.parametrize("bad", [[], {"beats": [1, 2, 3, 4]}])
def test_malformed_plan_is_repairable_validation_error(bad):
    from editorial import place_quote_ranges
    with pytest.raises(ValueError):
        place_quote_ranges(bad, [], 0, 120, 60)


def test_recovery_requires_all_four_complete_records():
    import json
    from editorial import decode_plan
    raw = json.dumps(valid_plan()) + " trailing model text"
    assert len(decode_plan(raw)["beats"]) == 4
    damaged = json.dumps({"title": "Test", "beats": valid_plan()["beats"][:3]})[:-2]
    with pytest.raises(ValueError, match="only 3 of four"):
        decode_plan(damaged)


def test_review_range_check_tolerates_only_serialization_rounding():
    from main import reviewed_ranges_match
    assert reviewed_ranges_match([(100.123, 107.777)], [(100.12300000001, 107.77700000001)])
    assert not reviewed_ranges_match([(100.13, 107.777)], [(100.123, 107.777)])
    assert not reviewed_ranges_match([], [(100, 107)])
