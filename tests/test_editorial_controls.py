import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import main
import metadata


class EditorialControlsTests(unittest.TestCase):
    def test_scene_selection_reaches_picker_without_prompt(self):
        class PickerReached(Exception):
            pass
        with tempfile.TemporaryDirectory() as folder:
            subtitles = Path(folder) / "movie.srt"
            subtitles.write_text("1\n00:00:00,000 --> 00:00:02,000\nA question.\n", encoding="utf-8")
            with patch.object(sys, "argv", ["main", "--video", __file__, "--selection", "scene", "--srt", str(subtitles), "--clip-dir", folder, "--no-bg-music"]), patch.object(main, "prompt_ranges", side_effect=AssertionError("unexpected prompt")), patch.object(main, "suggest_scenes", side_effect=PickerReached), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(PickerReached):
                    main.main()

    def test_approved_commentary_rejects_out_of_order_timing_before_render(self):
        with tempfile.TemporaryDirectory() as folder:
            approved = Path(folder) / "editorial.json"
            approved.write_text(json.dumps({"title": "Review", "description": "Analysis.", "tags": [], "commentary": [{"anchor": 8, "text": "First."}, {"anchor": 3, "text": "Second."}]}), encoding="utf-8")
            output = io.StringIO()
            with patch.object(sys, "argv", ["main", "--video", __file__, "--voice", "commentary", "--editorial-file", str(approved)]), contextlib.redirect_stderr(output):
                with self.assertRaises(SystemExit):
                    main.main()
            self.assertIn("anchors must increase", output.getvalue())

    def test_invalid_combinations_fail_before_prompt_or_generation(self):
        cases = [
            (["--selection", "scene"], "needs a subtitle"),
            (["--selection", "manual", "--voice", "storytelling"], "requires --selection script"),
            (["--selection", "script", "--voice", "none"], "requires --voice storytelling"),
            (["--selection", "script", "--voice", "storytelling"], "needs a subtitle"),
        ]
        for flags, message in cases:
            with self.subTest(flags=flags):
                output = io.StringIO()
                with patch.object(sys, "argv", ["main", "--video", __file__, "--no-srt"] + flags), patch.object(main, "prompt_ranges", side_effect=AssertionError("unexpected prompt")), contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    with self.assertRaises(SystemExit):
                        main.main()
                self.assertIn(message, output.getvalue())


    def test_approved_timing_rejects_overlap_and_window_overflow(self):
        from tts import validate_approved_timing
        from narration import NarrationBeat
        beat = NarrationBeat(2.0, 3.0, 2, "", "Original analysis")
        validate_approved_timing(beat, 2.0, 2.0, 0.0, 8.0)
        for duration, previous in ((3.0, 0.0), (2.0, 2.1)):
            with self.assertRaisesRegex(ValueError, "does not fit"):
                validate_approved_timing(beat, 2.0, duration, previous, 8.0)

    def test_original_commentary_mix_preserves_full_source_audio(self):
        import burn
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(burn, "_get_duration", return_value=2.0), patch.object(burn, "_run_ffmpeg") as run:
                burn.burn_subtitles(root / "in.mp4", root / "c.ass", root / "out.mp4", narration_audio=root / "voice.wav")
                command = run.call_args.args[0]
                graph = command[command.index("-filter_complex") + 1]
                self.assertNotIn("sidechaincompress", graph)
                self.assertIn("volume=0.04", graph)
                self.assertIn("[0:a]aresample=48000,aformat=channel_layouts=stereo[a0]", graph)

    def test_narrator_mix_ducks_source_and_can_mute_it(self):
        import burn
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source_volume in (0.22, 0.0):
                with patch.object(burn, "_get_duration", return_value=2.0), patch.object(burn, "_run_ffmpeg") as run:
                    burn.burn_subtitles(root / "in.mp4", root / "c.ass", root / "out.mp4",
                                        narration_audio=root / "voice.wav", narration_lead=True,
                                        dialogue_volume=source_volume)
                    command = run.call_args.args[0]
                    graph = command[command.index("-filter_complex") + 1]
                    self.assertIn("sidechaincompress", graph)
                    self.assertIn(f"volume={source_volume}", graph)

    def test_reviewed_script_preserves_copy_and_checks_source_windows(self):
        from tts import NarrationSegment
        segs = main.validate_approved_script([{"start": 20, "end": 30, "narration": "Reviewed analysis."}])
        spoken = [NarrationSegment("Reviewed analysis.", 0, 4)]
        self.assertEqual(segs[0]["narration"], "Reviewed analysis.")
        self.assertEqual(main.approved_script_ranges(segs, spoken, 100, [(20, 30)]), [(20, 24.8)])
        for movie_end, region in ((22, None), (100, [(25, 30)])):
            with self.assertRaises(ValueError):
                main.approved_script_ranges(segs, spoken, movie_end, region)
        with self.assertRaises(ValueError):
            main.approved_script_ranges(segs, [], 100)
        with self.assertRaises(ValueError):
            main.approved_script_ranges(segs, [NarrationSegment("Reviewed analysis.", 0, 11)], 100)

    def test_reviewed_script_rejects_invalid_or_reordered_entries(self):
        for lines in ([], [{"start": float("nan"), "end": 30, "narration": "Text"}],
                      [{"start": 20, "end": 19, "narration": "Text"}],
                      [{"start": 20, "end": 30, "narration": " "}],
                      [{"start": 20, "end": 30, "narration": "A"}, {"start": 10, "end": 18, "narration": "B"}]):
            with self.assertRaises(ValueError):
                main.validate_approved_script(lines)

    def test_narration_budget_scales_below_twelve_segments(self):
        from narrative import narration_budget
        count, words = narration_budget(90)
        self.assertLess(count, 12)
        self.assertLess(words, 225)
        self.assertGreater(narration_budget(180)[1], words)
        for bad in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                narration_budget(bad)

    def test_storytelling_short_duration_guard(self):
        main.check_storytelling_duration(89.9, 90, "vertical")
        main.check_storytelling_duration(199, 120, "landscape")
        for duration, target in ((91, 90), (181, 240)):
            with self.assertRaisesRegex(ValueError, "no video was rendered"):
                main.check_storytelling_duration(duration, target, "vertical")

    def test_hook_retries_relative_timestamps_and_stays_in_scene(self):
        import hook
        invalid = json.dumps({"hook": [{"start": "0:00", "end": "0:03", "line": "Wrong movie moment."}]})
        valid = json.dumps({"hook": [{"start": "16:12", "end": "16:15", "line": "Where are you taking me?"}]})
        with tempfile.TemporaryDirectory() as folder:
            subtitles = Path(folder) / "movie.srt"
            subtitles.write_text("1\n00:16:12,000 --> 00:16:15,000\nWhere are you taking me?\n", encoding="utf-8")
            for retry_response in (valid, invalid):
                with self.subTest(retry_response=retry_response), patch.object(hook, "fetch_movie_context", return_value="Film"), patch.object(hook, "_call_ollama", side_effect=[invalid, retry_response]) as call:
                    picks = hook.suggest_hook(subtitles, region=[(970, 980)])
                    self.assertEqual(call.call_count, 2)
                    if retry_response == valid:
                        self.assertEqual(len(picks), 1)
                        self.assertGreaterEqual(picks[0]["start"], 970)
                        self.assertLessEqual(picks[0]["end"], 980)
                    else:
                        self.assertEqual(picks, [])

    def test_long_form_metadata_includes_original_perspective(self):
        response = json.dumps({"title": "Test", "description": "A scene analysis.", "tags": []})
        with patch.object(metadata, "fetch_movie_context", return_value="A film."), patch.object(metadata, "_call_ollama", return_value=response) as call:
            metadata.generate_metadata("dialogue", "Film", content_format="long-form", editorial_context="Expertise versus control")
        prompt = call.call_args.args[1]
        self.assertNotIn("YouTube Short", prompt)
        self.assertIn("Expertise versus control", prompt)


if __name__ == "__main__":
    unittest.main()
