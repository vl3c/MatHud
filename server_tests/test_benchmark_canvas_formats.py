"""Tests for scripts/benchmark_canvas_formats.py (canvas-format comprehension benchmark).

Covers question generation and ground truths, the pure grading functions, that
prompts are the provider's own rendering per format, the dry run with the
network blocked, and the live paths (OpenRouter and LocalAgent) with a mocked
OpenAI client.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import patch

from static.ai_model import AIModel
from static.canvas_state_formatter import render_state, render_update
from static.providers.local.local_agent_api import LocalAgentAPI
from static.providers.openrouter_api import OpenRouterAPI

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "benchmark_canvas_formats.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_canvas_formats", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


bench = _load_script()

SCENES = bench.load_scenes()
QUESTIONS = bench.build_questions(SCENES)
SCENE_BY_NAME = {scene.name: scene for scene in SCENES}


def question(qid: str) -> Any:
    return next(q for q in QUESTIONS if q.qid == qid)


@contextlib.contextmanager
def network_blocked() -> Iterator[List[Any]]:
    """Refuse every socket connection and record the attempts."""
    attempts: List[Any] = []

    def refuse(self: socket.socket, address: Any) -> None:
        attempts.append(address)
        raise OSError(f"network blocked in test: {address}")

    with patch.object(socket.socket, "connect", refuse), patch.object(socket.socket, "connect_ex", refuse):
        yield attempts


class TestQuestions(unittest.TestCase):
    def test_every_scene_has_questions_with_ground_truths(self) -> None:
        for scene in SCENES:
            scene_questions = [q for q in QUESTIONS if q.scene == scene.name]
            self.assertGreaterEqual(len(scene_questions), 8, scene.name)
            for q in scene_questions:
                self.assertTrue(q.text.strip(), q.qid)
                self.assertIn(q.kind, bench.ANSWER_KINDS)
                self.assertIsNotNone(q.expected, q.qid)
                if q.kind in ("numbers", "name", "expression", "edge"):
                    self.assertTrue(q.expected, q.qid)

    def test_question_ids_are_unique(self) -> None:
        ids = [q.qid for q in QUESTIONS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_ground_truths_come_from_the_fixtures(self) -> None:
        self.assertAlmostEqual(question("tc_len_bc").expected, math.sqrt(18))
        self.assertAlmostEqual(question("tc_area_abc").expected, 6.0)
        self.assertEqual(question("tc_largest_angle").expected, ["A"])
        self.assertEqual(question("tc_on_circle").expected, [])
        self.assertAlmostEqual(question("mm_angle_bac").expected, math.degrees(math.atan2(3, 4)))
        self.assertEqual(question("mm_right_angle").expected, ["B"])
        self.assertEqual(question("mm_closest_to_h").expected, ["D"])
        self.assertEqual(question("wg_edge_count").expected, 12)
        self.assertEqual(question("wg_lightest_edge_d").expected, ["D", "E"])
        self.assertEqual(question("wg_shortest_a_f").expected, 12)
        self.assertEqual(question("wg_neighbours_f").expected, ["D", "E", "G"])
        self.assertEqual(question("rd_named_f_count").expected, 25)
        self.assertEqual(question("rd_bar_max").expected, ["Fri"])
        self.assertAlmostEqual(question("rd_fit_slope").expected, 1.8595995550611797)

    def test_closest_point_question_excludes_labels(self) -> None:
        self.assertIn("named point (not label)", question("mm_closest_to_h").text)

    def test_change_questions_follow_the_fixture_diff(self) -> None:
        self.assertEqual(question("ch_removed").expected, ["EF"])
        self.assertEqual(question("ch_moved_point").expected, ["I"])
        self.assertEqual(question("ch_old_position").expected, [-1.0, 0.5])
        self.assertEqual(question("ch_new_position").expected, [-1.0, -0.5])
        self.assertEqual(question("ch_added_name").expected, ["A(5)"])
        self.assertEqual(question("ch_on_new_circle").expected, ["C"])
        self.assertEqual(question("ch_segments_now").expected, ["AB", "AC", "BC", "GH"])
        self.assertIn("EF", question("ch_removed").vocabulary)


class TestGrading(unittest.TestCase):
    def test_extract_answer_takes_the_last_answer_line(self) -> None:
        reply = "<think>Answer: 1</think>Working...\nAnswer: 3\nso\n**Answer:** `4.24`"
        self.assertEqual(bench.extract_answer(reply), ("4.24", True))
        self.assertEqual(bench.extract_answer("It is 7.\n\n"), ("It is 7.", False))
        self.assertEqual(bench.extract_answer(""), ("", False))

    def test_extract_answer_accepts_final_answer_lines_and_boxed_values(self) -> None:
        self.assertEqual(bench.extract_answer("Work.\nFinal Answer: B"), ("B", True))
        self.assertEqual(bench.extract_answer("**Final answer:** B"), ("B", True))
        self.assertEqual(bench.extract_answer("\\boxed{D}"), ("D", False))
        self.assertEqual(bench.extract_answer("Answer: $\\boxed{\\frac{1}{2}}$"), ("\\frac{1}{2}", True))
        self.assertTrue(bench.grade(question("mm_right_angle"), bench.extract_answer("Final Answer: B")[0]))
        self.assertTrue(bench.grade(question("mm_closest_to_h"), bench.extract_answer("\\boxed{D}")[0]))

    def test_numbers_use_a_tolerance(self) -> None:
        q = question("tc_len_bc")
        self.assertTrue(bench.grade(q, "4.24"))
        self.assertTrue(bench.grade(q, "3√2 ≈ 4.243"))
        self.assertFalse(bench.grade(q, "4.3"))
        self.assertFalse(bench.grade(q, "unknown"))
        self.assertTrue(bench.numbers_close(0.005, 0.0))
        self.assertTrue(bench.grade(question("wg_shortest_a_f"), "12 (A-C-E-F)"))

    def test_numbers_rounded_to_whole_numbers_are_wrong(self) -> None:
        self.assertAlmostEqual(question("rd_view_top").expected, 20.14256619144603)
        self.assertFalse(bench.grade(question("rd_view_top"), "20"))
        self.assertTrue(bench.grade(question("rd_view_top"), "20.14"))
        self.assertFalse(bench.grade(question("wg_point_b"), "140, 140"))
        self.assertTrue(bench.grade(question("wg_point_b"), "140.856, 140.856"))
        self.assertFalse(bench.grade(question("wg_distance_ac"), "280"))
        self.assertTrue(bench.grade(question("wg_distance_ac"), "281.7"))
        self.assertTrue(bench.grade(question("mm_point_j"), "3.33, 1.1"))  # 2 decimals of 10/3

    def test_ordered_numbers(self) -> None:
        q = question("ch_old_position")
        self.assertTrue(bench.grade(q, "(-1, 0.5)"))
        self.assertTrue(bench.grade(q, "x = −1, y = 0.5"))
        self.assertFalse(bench.grade(q, "(0.5, -1)"))
        self.assertFalse(bench.grade(q, "-1"))

    def test_new_position_takes_the_part_after_an_arrow(self) -> None:
        q = question("ch_new_position")
        self.assertTrue(bench.grade(q, "(-1, 0.5) -> (-1, -0.5)"))
        self.assertTrue(bench.grade(q, "(-1, 0.5) → (-1, -0.5)"))
        self.assertTrue(bench.grade(q, "(-1, -0.5)"))
        self.assertFalse(bench.grade(q, "(-1, -0.5) -> (-1, 0.5)"))

    def test_names_are_case_insensitive(self) -> None:
        self.assertTrue(bench.grade(question("mm_e1_color"), "Purple."))
        self.assertTrue(bench.grade(question("mm_ac_label"), '"diag"'))
        self.assertTrue(bench.grade(question("tc_largest_angle"), "Vertex a (71.57 degrees)"))
        self.assertTrue(bench.grade(question("ch_added_name"), "circle A(5)"))
        self.assertTrue(bench.grade(question("rd_bar_max"), "Fri (22)"))
        self.assertFalse(bench.grade(question("ch_added_name"), "A(3)"))
        self.assertFalse(bench.grade(question("tc_largest_angle"), "B"))

    def test_sets_are_order_insensitive(self) -> None:
        q = question("ch_segments_now")
        self.assertTrue(bench.grade(q, "GH, AB, ac and BC"))
        self.assertFalse(bench.grade(q, "AB, BC, AC, EF, GH"))
        circle = question("tc_on_circle")
        self.assertTrue(bench.grade(circle, "None"))
        self.assertFalse(bench.grade(circle, "B"))
        # The article "a" is not point A.
        self.assertTrue(bench.grade(question("ch_on_new_circle"), "Only C lies on a circle of radius 5"))
        self.assertEqual(bench.parse_name_set("points D, E and G", ["D", "E", "G", "AB"]), {"D", "E", "G"})

    def test_set_answers_end_at_an_explanation(self) -> None:
        self.assertTrue(bench.grade(question("wg_neighbours_f"), "D, E, G (via DF, EF, FG)"))
        self.assertTrue(bench.grade(question("ch_segments_now"), "AB, BC, AC, GH (EF was removed)"))
        self.assertTrue(bench.grade(question("ch_segments_now"), "AB, BC, AC, GH; EF is gone"))
        self.assertTrue(bench.grade(question("ch_on_new_circle"), "C (I is inside)"))
        self.assertTrue(bench.grade(question("ch_on_new_circle"), "C because |AC| = 5"))
        self.assertTrue(bench.grade(question("ch_on_new_circle"), "C lies on A(5)"))
        self.assertFalse(bench.grade(question("ch_on_new_circle"), "C, I"))

    def test_set_vocabulary_holds_only_the_kind_asked_for(self) -> None:
        points = question("ch_on_new_circle").vocabulary
        self.assertIn("C", points)
        self.assertNotIn("A(5)", points)
        segments = question("ch_segments_now").vocabulary
        self.assertIn("EF", segments)
        self.assertNotIn("A", segments)

    def test_edges_and_expressions(self) -> None:
        edge = question("wg_lightest_edge_d")
        self.assertTrue(bench.grade(edge, "E-D"))
        self.assertTrue(bench.grade(edge, "DE (weight 4)"))
        self.assertFalse(bench.grade(edge, "B-D"))
        expression = question("mm_f_expression")
        self.assertTrue(bench.grade(expression, "f(x) = x**2 - 1"))
        self.assertTrue(bench.grade(expression, "x² − 1"))
        self.assertFalse(bench.grade(expression, "x^2 + 1"))


class PromptEnv(unittest.TestCase):
    """Builds prompts the way the script does, with no key or .env access."""

    def setUp(self) -> None:
        self._env = patch.dict(os.environ, {"OPENROUTER_API_KEY": ""})
        self._env.start()
        self.addCleanup(self._env.stop)

    def request(self, provider: str, fmt: str, qid: str, model: str = "deepseek/deepseek-v4.1-flash") -> Any:
        requests, _ = bench.build_requests(provider, [model], [fmt], SCENES, [question(qid)], None, True)
        return requests[0]

    def own_openrouter_content(self, fmt: str, prompt: str) -> Any:
        with patch.dict(os.environ, {"MATHUD_CANVAS_FORMAT": fmt, "OPENROUTER_API_KEY": "test-key"}):
            api = OpenRouterAPI(model=AIModel.from_identifier("deepseek/deepseek-v4.1-flash"), tool_mode="search")
            return api._prepare_message_content(prompt)


class TestPrompts(PromptEnv):
    def test_text_and_json_prompts_differ_and_hold_the_apps_canvas(self) -> None:
        q = question("tc_len_bc")
        state = SCENE_BY_NAME["triangle_circle"].state
        user_text = bench.question_user_text(q, False)
        prompt = bench.user_prompt_json(user_text, state, "deepseek/deepseek-v4.1-flash")
        text_request = self.request("openrouter", "text", "tc_len_bc")
        json_request = self.request("openrouter", "json", "tc_len_bc")
        text_content = text_request.messages[-1]["content"]
        json_content = json_request.messages[-1]["content"]
        self.assertNotEqual(text_content, json_content)
        self.assertEqual(text_content, self.own_openrouter_content("text", prompt))
        self.assertIn(render_state(state, "text", 4000), text_content)
        self.assertEqual(json_content, self.own_openrouter_content("json", prompt))
        self.assertEqual(json.loads(json_content)["canvas_state"], state)
        self.assertNotIn("_p1_coords", text_content)

    def test_system_prompt_matches_the_format_and_role(self) -> None:
        for fmt in ("text", "json"):
            with patch.dict(os.environ, {"MATHUD_CANVAS_FORMAT": fmt, "OPENROUTER_API_KEY": "test-key"}):
                expected = OpenRouterAPI(
                    model=AIModel.from_identifier("deepseek/deepseek-v4.1-flash"), tool_mode="search"
                ).messages[0]
            self.assertEqual(self.request("openrouter", fmt, "tc_len_bc").messages[0], expected)
            self.assertEqual(expected["role"], "system")

    def test_no_tools_and_openrouter_sampling_parameters(self) -> None:
        request = self.request("openrouter", "text", "tc_len_bc")
        self.assertEqual(request.request_kwargs, {"max_tokens": 16000})
        local = self.request("local", "text", "tc_len_bc", model="local-model")
        self.assertEqual(local.request_kwargs, {"max_tokens": 16000, "temperature": 0.2})

    def test_change_scene_text_reports_the_canvas_changes(self) -> None:
        scene = SCENE_BY_NAME[bench.CHANGE_SCENE]
        messages = self.request("openrouter", "text", "ch_old_position").messages
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool", "tool", "tool"])
        self.assertIn(render_state(scene.state, "text", 4000), messages[1]["content"])
        delta = render_update(scene.state, scene.after, "text", 4000)
        self.assertTrue(delta.startswith("[canvas changes]"))
        self.assertTrue(messages[-1]["content"].endswith(delta))
        self.assertIn("- EF (segment) removed", delta)

    def test_change_scene_json_sends_no_canvas_after_the_tool_batch(self) -> None:
        messages = self.request("openrouter", "json", "ch_old_position").messages
        serialized = json.dumps(messages[1:])  # the system prompt names get_current_canvas_state
        self.assertNotIn("canvas_state", serialized)
        self.assertNotIn("Points", serialized)
        self.assertNotIn("[canvas changes]", serialized)
        self.assertIn("update_point", serialized)

    def test_local_prompts_use_the_local_provider(self) -> None:
        q = question("tc_len_bc")
        state = SCENE_BY_NAME["triangle_circle"].state
        prompt = bench.user_prompt_json(bench.question_user_text(q, False), state, "local-model")
        for fmt in ("text", "json"):
            request = self.request("local", fmt, "tc_len_bc", model="local-model")
            with patch.dict(os.environ, {"MATHUD_CANVAS_FORMAT": fmt}):
                expected = LocalAgentAPI(model=AIModel.from_identifier("local-model"))._parse_and_prepare_message(
                    prompt
                )
            self.assertEqual(request.messages[-1], expected)
        json_request = self.request("local", "json", "tc_len_bc", model="local-model")
        self.assertIn("[Canvas: 3 Points, 3 Segments, 1 Triangles, 1 Circles]", json_request.messages[-1]["content"])

    def test_canvas_budget_option_reaches_the_renderer(self) -> None:
        requests, _ = bench.build_requests(
            "local", ["local-model"], ["text"], SCENES, [question("rd_point_count")], 150, True
        )
        self.assertIn("omitted", requests[0].messages[-1]["content"])
        self.assertIsNone(os.environ.get("MATHUD_CANVAS_BUDGET_TOKENS"))


class FakeCompletions:
    def __init__(self, reply: str, timings: Optional[Dict[str, float]] = None) -> None:
        self.reply = reply
        self.timings = timings
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=50, total_tokens=1050),
        )
        if self.timings is not None:
            response.timings = self.timings
        return response


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)
        self.options: List[Dict[str, Any]] = []

    def with_options(self, **kwargs: Any) -> "FakeClient":
        self.options.append(kwargs)
        return self


def run_main(args: List[str]) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = bench.main(args)
    return code, out.getvalue()


class TestDryRun(unittest.TestCase):
    def test_dry_run_builds_everything_without_network_or_env_files(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}),
            patch.object(bench.env_config, "load_env_files", side_effect=AssertionError("no .env in a dry run")),
            network_blocked() as attempts,
        ):
            code, output = run_main(["--formats", "json", "text", "--dry-run", "--out", tmp])
            self.assertEqual(code, 0)
            self.assertEqual(attempts, [])
            self.assertIn(f"Planned requests: {2 * 2 * len(QUESTIONS)}", output)
            self.assertIn("Estimated total cost: $", output)
            report = json.loads((Path(tmp) / "dry_run.json").read_text(encoding="utf-8"))
            self.assertEqual({row["format"] for row in report["rows"]}, {"json", "text"})
            prompts = list((Path(tmp) / "prompts").rglob("*.json"))
            self.assertEqual(len(prompts), 2 * 2 * len(QUESTIONS))
            self.assertEqual(os.environ.get("OPENROUTER_API_KEY"), "")

    def test_local_dry_run_skips_model_discovery(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(LocalAgentAPI, "fetch_models", side_effect=AssertionError("no discovery in a dry run")),
            network_blocked() as attempts,
        ):
            code, output = run_main(["--provider", "local", "--dry-run", "--scenes", "triangle_circle", "--out", tmp])
            self.assertEqual(code, 0)
            self.assertEqual(attempts, [])
            self.assertIn("Models: local-model", output)


class TestLiveOpenRouterMocked(unittest.TestCase):
    def setUp(self) -> None:
        self.completions = FakeCompletions("The path A-C-E-F.\nAnswer: 12")
        self.client = FakeClient(self.completions)
        for target in (
            patch("static.providers.openrouter_api.OpenAI", return_value=self.client),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            patch.object(bench.env_config, "load_env_files"),
        ):
            target.start()
            self.addCleanup(target.stop)

    def test_live_run_writes_results_and_prints_the_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, network_blocked() as attempts:
            code, output = run_main(
                ["--models", "deepseek/deepseek-v4.1-flash", "--scenes", "weighted_graph", "--out", tmp]
            )
            self.assertEqual(code, 0)
            self.assertEqual(attempts, [])
            results = json.loads((Path(tmp) / "results.json").read_text(encoding="utf-8"))
            summary_md = (Path(tmp) / "summary.md").read_text(encoding="utf-8")
        self.assertIn("## Overall", output)
        self.assertIn("| deepseek/deepseek-v4.1-flash | json |", summary_md)
        self.assertIn("| deepseek/deepseek-v4.1-flash | text |", summary_md)
        self.assertIn("## Wrong answers", summary_md)
        per_scene = len([q for q in QUESTIONS if q.scene == "weighted_graph"])
        self.assertEqual(len(results["results"]), 2 * per_scene)
        by_id = {(r["format"], r["qid"]): r for r in results["results"]}
        self.assertTrue(by_id[("text", "wg_shortest_a_f")]["correct"])
        self.assertTrue(by_id[("json", "wg_edge_count")]["correct"])  # 12 edges as well
        self.assertFalse(by_id[("text", "wg_weight_bd")]["correct"])
        row = results["summary"][0]
        self.assertEqual(row["mean_prompt_tokens"], 1000)
        self.assertEqual(row["errors"], 0)
        self.assertAlmostEqual(row["cost_usd"], per_scene * (1000 * 0.099 + 50 * 0.60) / 1_000_000)
        for call in self.completions.calls:
            self.assertNotIn("tools", call)
            self.assertNotIn("temperature", call)
            self.assertEqual(call["model"], "deepseek/deepseek-v4.1-flash")
            self.assertEqual(call["max_tokens"], 16000)
        self.assertEqual(self.client.options[0], {"timeout": 180.0})

    def test_request_errors_are_recorded_not_raised(self) -> None:
        self.completions.create = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as tmp:
            code, _ = run_main(["--models", "m/x", "--formats", "text", "--scenes", "triangle_circle", "--out", tmp])
            results = json.loads((Path(tmp) / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertTrue(all(r["error"] == "RuntimeError: boom" and not r["correct"] for r in results["results"]))
        self.assertEqual(results["summary"][0]["errors"], len(results["results"]))

    def test_max_requests_aborts_before_sending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, output = run_main(["--max-requests", "10", "--out", tmp])
        self.assertEqual(code, 2)
        self.assertIn("Nothing was sent", output)
        self.assertEqual(self.completions.calls, [])

    def test_missing_key_fails_clearly(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}), tempfile.TemporaryDirectory() as tmp:
            code, output = run_main(["--out", tmp])
        self.assertEqual(code, 2)
        self.assertIn("OPENROUTER_API_KEY is not set", output)
        self.assertEqual(self.completions.calls, [])


class TestLiveLocalMocked(unittest.TestCase):
    def test_local_run_discovers_models_and_records_timings(self) -> None:
        completions = FakeCompletions(
            "Answer: 6", timings={"predicted_per_second": 42.5, "predicted_n": 50, "prompt_per_second": 900.0}
        )
        client = FakeClient(completions)
        models_response = SimpleNamespace(
            status_code=200, raise_for_status=lambda: None, json=lambda: {"data": [{"id": "qwen-test"}]}
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_AGENT_BASE_URL": "http://127.0.0.1:9999"}),
            patch.object(bench.env_config, "load_env_files"),
            patch("requests.get", return_value=models_response) as get,
            patch("openai.OpenAI", return_value=client) as openai_cls,
        ):
            code, output = run_main(["--provider", "local", "--scenes", "triangle_circle", "--out", tmp])
            results = json.loads((Path(tmp) / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(get.call_args.args[0], "http://127.0.0.1:9999/v1/models")
        self.assertEqual(openai_cls.call_args.kwargs["base_url"], "http://127.0.0.1:9999/v1")
        self.assertIn("Models: qwen-test", output)
        self.assertEqual(client.options[0], {"timeout": 900.0})
        for call in completions.calls:
            self.assertEqual(call["model"], "qwen-test")
            self.assertEqual(call["temperature"], 0.2)
            self.assertNotIn("tools", call)
        first = results["results"][0]
        self.assertEqual(first["output_tokens_per_s"], 42.5)
        self.assertEqual(first["server_timings"]["predicted_n"], 50)
        self.assertEqual(results["summary"][0]["mean_output_tokens_per_s"], 42.5)
        self.assertIsNone(results["summary"][0]["cost_usd"])
        area = next(r for r in results["results"] if r["qid"] == "tc_area_abc" and r["format"] == "text")
        self.assertTrue(area["correct"])
        text_prompt = next(
            c["messages"][-1]["content"]
            for c in completions.calls
            if c["messages"][-1]["content"].startswith("<canvas>")
        )
        self.assertIn("A(3) = Circle(center A, r 3)", text_prompt)


if __name__ == "__main__":
    unittest.main()
