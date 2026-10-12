import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import main
import outro
from tts import NarrationSegment


def test_brand_is_reloaded_and_avatar_is_optional(tmp_path, monkeypatch):
    brand_path = tmp_path / "assets" / "brand" / "brand.json"
    brand_path.parent.mkdir(parents=True)
    config = json.loads(outro.BRAND_PATH.read_text(encoding="utf-8"))
    config["watermark_text"] = "CUSTOM BRAND"
    brand_path.write_text(json.dumps(config), encoding="utf-8")
    brand = outro.load_brand(brand_path)
    assert brand["avatar_path"] == str(tmp_path / "assets" / "brand" / "avatar.png")
    assert not Path(brand["avatar_path"]).exists()
    monkeypatch.setattr(outro, "load_brand", lambda: brand)
    from burn import get_watermarks
    assert get_watermarks() == {"mv-edits": "CUSTOM BRAND"}


def test_spoken_coverage_unions_intervals_and_excludes_bylines():
    speech = [NarrationSegment("a", -1, 2), NarrationSegment("b", 1, 4),
              NarrationSegment("c", 8, 20)]
    assert main.narration_coverage(speech, 10) == 60
    assert main.narration_coverage(speech, 10, byline_only=True) == 0
    assert main.narration_coverage([], 0) == 0


@pytest.mark.parametrize("duration", [0, 1.49, 2.51, float("nan"), float("inf")])
def test_invalid_card_duration_fails_before_video_prompt(monkeypatch, duration):
    monkeypatch.setattr(sys, "argv", ["main", "--end-card-duration", str(duration)])
    monkeypatch.setattr(main, "prompt_video_path", lambda: pytest.fail("Unexpected prompt"))
    with pytest.raises(SystemExit) as error:
        main.main()
    assert error.value.code == 2


@pytest.mark.parametrize("colour", ["red", "#abcdef0", None])
def test_bad_brand_colour_reports_configuration_error(tmp_path, colour):
    config = json.loads(outro.BRAND_PATH.read_text(encoding="utf-8"))
    config["accent_colour"] = colour
    path = tmp_path / "brand.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="accent_colour"):
        outro.load_brand(path)


@pytest.mark.parametrize("has_avatar", [False, True])
def test_card_uses_configured_identity_without_prompt_invention(monkeypatch, tmp_path, has_avatar):
    import subprocess
    brand = outro.load_brand()
    brand["display_name"] = "A channel's identity: 100%"
    brand["subscribe_prompt"] = "Subscribe"
    avatar = tmp_path / "avatar.png"
    if has_avatar:
        avatar.write_bytes(b"image fixture")
    brand["avatar_path"] = str(avatar)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        graph = command[command.index("-filter_complex") + 1]
        assert "expansion=none" in graph
        assert ("[2:v]" in graph) == has_avatar
        text_files = list(Path(__import__("tempfile").gettempdir()).glob("brand-card-*/text-*.txt"))
        assert any(p.read_text(encoding="utf-8") == brand["display_name"] for p in text_files)
    monkeypatch.setattr(subprocess, "run", run)
    outro.build_brand_end_card(tmp_path / "card.mp4", brand=brand)
    assert len(commands) == 1


def test_unreadable_avatar_retries_without_artwork(monkeypatch, tmp_path):
    import subprocess
    brand = outro.load_brand()
    avatar = tmp_path / "bad.png"
    avatar.write_bytes(b"not an image")
    brand["avatar_path"] = str(avatar)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, "run", run)
    outro.build_brand_end_card(tmp_path / "card.mp4", brand=brand)
    assert len(commands) == 2
    assert str(avatar) not in commands[-1]


@pytest.mark.parametrize("shape,enabled,fail_card", [("vertical", True, False), ("square", True, False),
                        ("landscape", True, False), ("square", False, False), ("square", True, True)])
def test_editorial_render_delivers_card_and_final_record_without_llm(tmp_path, monkeypatch, shape, enabled, fail_card):
    """Small real encode checks integration that mocked filter tests cannot."""
    import subprocess
    import shutil
    from ffutil import get_duration, get_resolution
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe required for encode integration")
    movie = tmp_path / "fixture.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=24:d=3",
                    "-f", "lavfi", "-i", "sine=frequency=200:duration=3", "-c:v", "libx264",
                    "-preset", "ultrafast", "-c:a", "aac", str(movie)], check=True, capture_output=True)
    subtitles = tmp_path / "fixture.srt"
    subtitles.write_text("1\n00:00:00,000 --> 00:00:02,800\nA source line.\n", encoding="utf-8")
    editorial = tmp_path / "editorial.json"
    editorial.write_text(json.dumps({"title": "Brand integration", "description": "Test only.", "tags": [],
                                    "commentary": [{"anchor": .2, "text": "A brief text byline."}]}), encoding="utf-8")
    folder = tmp_path / "render"
    args = ["main", "--video", str(movie), "--ranges", "0:00-0:03", "--voice", "commentary",
            "--shape", shape, "--byline-only", "--editorial-file", str(editorial),
            "--srt", str(subtitles), "--clip-dir", str(folder), "--no-bg-music", "--no-thumbnail",
            "--no-snap-speech", "--no-edit-credits", "--watermark", "none", "--no-caption-shuffle"]
    if not enabled:
        args += ["--no-end-card"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(main, "generate_metadata", lambda *a, **k: pytest.fail("LLM must not run"))
    monkeypatch.setattr(main, "synthesize_narration", lambda *a, **k: pytest.fail("Bylines must not call TTS"))
    if fail_card:
        def fail(*a, **k):
            raise RuntimeError("Card fixture failed")
        monkeypatch.setattr(main, "build_brand_end_card", fail)
        with pytest.raises(SystemExit) as error:
            main.main()
        assert error.value.code == 2
    else:
        main.main()
    final = folder / "clip_final.mp4"
    record = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))["render"]
    assert get_resolution(final) == {"vertical": (1080, 1920), "square": (1080, 1080),
                                     "landscape": (1920, 1080)}[shape]
    assert record["title"] == "Brand integration"
    assert record["end_card_requested"] is enabled
    assert record["end_card"] is (enabled and not fail_card)
    assert record["narration_coverage_pct"] == 0
    assert record["final_duration"] == get_duration(final)
    # Body-only mastering may leave a short AAC tail; don't claim an exact cap.
    assert abs(record["final_duration"] - (5 if enabled and not fail_card else 3)) < .15
    assert (folder / "end_card.mp4").exists() is (enabled and not fail_card)


@pytest.mark.parametrize("enabled,duration,status", [(True, 1.5, 200), (False, 2.5, 200),
                         (True, 0, 400), (True, "bad", 400), (True, float("nan"), 400)])
def test_web_end_card_controls_route_without_ollama(monkeypatch, enabled, duration, status):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))
    import app
    arguments = []
    async def stream(args):
        arguments.extend(args)
    async def forbidden():
        pytest.fail("Approved render must not start Ollama")
    monkeypatch.setattr(app, "_stream_run", stream)
    monkeypatch.setattr(app, "_ensure_ollama", forbidden)
    # Mock scheduling at the external launch boundary. These request/launch
    # stubs complete immediately; no socket-dependent Windows loop is needed.
    def immediate(coroutine):
        try:
            coroutine.send(None)
        except StopIteration as completed:
            return completed.value
        finally:
            coroutine.close()
        pytest.fail("Unexpected asynchronous I/O in isolated request test")
    monkeypatch.setattr(app.asyncio, "create_task", immediate)
    class Request:
        app = {}
        async def json(self):
            return {"video": __file__, "ranges": "0:00-0:03", "selection": "manual",
                    "voice": "commentary", "editorial_file": __file__,
                    "end_card": enabled, "end_card_duration": duration}
    response = immediate(app.start_run(Request()))
    assert response.status == status
    if status == 200:
        assert ("--end-card" if enabled else "--no-end-card") in arguments
        assert arguments[arguments.index("--end-card-duration") + 1] == str(float(duration))
    else:
        assert not arguments
