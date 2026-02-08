from __future__ import annotations

import unittest

from .simple_mock import SimpleMock
from drawables.point import Point
from drawables.segment import Segment
from utils.math_utils import MathUtils
from workspace_manager import WorkspaceManager


class TestWorkspaceSegmentPersistence(unittest.TestCase):
    def test_segment_state_preserves_endpoint_order(self) -> None:
        point_o = Point(-70.0, -200.0, name="O")
        point_k = Point(-20.0, -180.0, name="K")
        segment = Segment(point_o, point_k)

        state = segment.get_state()

        self.assertEqual(state["args"]["p1"], "O")
        self.assertEqual(state["args"]["p2"], "K")

    def test_legacy_segment_state_restores_original_orientation(self) -> None:
        point_o = Point(-70.0, -200.0, name="O")
        point_k = Point(-20.0, -180.0, name="K")
        points = {"O": point_o, "K": point_k}

        captured_segments = []

        def get_point_by_name(name: str) -> Point | None:
            return points.get(name)

        def get_point(x: float, y: float) -> Point | None:
            for point in points.values():
                if MathUtils.point_matches_coordinates(point, x, y):
                    return point
            return None

        def create_segment(x1: float, y1: float, x2: float, y2: float, name: str = "", **kwargs: object) -> SimpleMock:
            captured_segments.append((x1, y1, x2, y2, name, kwargs))
            return SimpleMock()

        canvas = SimpleMock(
            get_point_by_name=get_point_by_name,
            get_point=get_point,
            create_segment=create_segment,
        )

        manager = WorkspaceManager(canvas)
        legacy_state = {
            "Segments": [
                {
                    "name": "OK",
                    "args": {
                        "p1": "K",
                        "p2": "O",
                        "p1_coords": [-70.0, -200.0],
                        "p2_coords": [-20.0, -180.0],
                    },
                }
            ]
        }

        manager._create_segments(legacy_state)

        self.assertEqual(len(captured_segments), 1)
        x1, y1, x2, y2, name, kwargs = captured_segments[0]
        self.assertEqual((x1, y1), (-70.0, -200.0))
        self.assertEqual((x2, y2), (-20.0, -180.0))
        self.assertEqual(name, "OK")
        self.assertEqual(kwargs.get("label_text"), "")
        self.assertFalse(bool(kwargs.get("label_visible", False)))


class TestWorkspaceLabelRestore(unittest.TestCase):
    def test_create_labels_restores_visible_scale_and_render_mode(self) -> None:
        created_labels = []

        def create_label(x: float, y: float, text: str, name: str = "", **kwargs: object) -> SimpleMock:
            label = SimpleMock(update_reference_scale=SimpleMock())
            created_labels.append((x, y, text, name, kwargs, label))
            return label

        canvas = SimpleMock(create_label=create_label)
        manager = WorkspaceManager(canvas)

        state = {
            "Labels": [
                {
                    "name": "lbl_A",
                    "args": {
                        "position": {"x": 1.0, "y": 2.0},
                        "text": "hello",
                        "visible": False,
                        "reference_scale_factor": 2.5,
                        "render_mode": {
                            "kind": "screen_offset",
                            "text_format": "text_only",
                            "offset_from_point_radius": True,
                        },
                    },
                }
            ]
        }

        manager._create_labels(state)

        self.assertEqual(len(created_labels), 1)
        x, y, text, name, kwargs, label = created_labels[0]
        self.assertEqual((x, y), (1.0, 2.0))
        self.assertEqual(text, "hello")
        self.assertEqual(name, "lbl_A")

        self.assertFalse(bool(getattr(label, "visible", True)))
        label.update_reference_scale.assert_called_once()
        args, _ = label.update_reference_scale.calls[0]
        self.assertEqual(args[0], 2.5)

        render_mode = getattr(label, "render_mode", None)
        self.assertEqual(getattr(render_mode, "kind", None), "screen_offset")

    def test_create_segments_restores_embedded_label_fields(self) -> None:
        points = {"A": Point(0.0, 0.0, name="A"), "B": Point(10.0, 0.0, name="B")}

        def get_point_by_name(name: str) -> Point | None:
            return points.get(name)

        def get_point(x: float, y: float) -> Point | None:
            for point in points.values():
                if MathUtils.point_matches_coordinates(point, x, y):
                    return point
            return None

        embedded_label = SimpleMock(update_font_size=SimpleMock(), update_rotation=SimpleMock())
        segment_obj = SimpleMock(label=embedded_label)

        def create_segment(x1: float, y1: float, x2: float, y2: float, name: str = "", **kwargs: object) -> SimpleMock:
            return segment_obj

        canvas = SimpleMock(
            get_point_by_name=get_point_by_name,
            get_point=get_point,
            create_segment=create_segment,
        )

        manager = WorkspaceManager(canvas)
        state = {
            "Segments": [
                {
                    "name": "AB",
                    "args": {
                        "p1": "A",
                        "p2": "B",
                        "p1_coords": [0.0, 0.0],
                        "p2_coords": [10.0, 0.0],
                        "label": {
                            "text": "mid",
                            "visible": True,
                            "font_size": 18.0,
                            "rotation_degrees": 30.0,
                            "render_mode": {"kind": "screen_offset", "text_format": "text_only"},
                        },
                    },
                }
            ]
        }

        manager._create_segments(state)

        embedded_label.update_font_size.assert_called_once()
        embedded_label.update_rotation.assert_called_once()
        render_mode = getattr(embedded_label, "render_mode", None)
        self.assertEqual(getattr(render_mode, "kind", None), "screen_offset")


class TestWorkspaceManagerHelperMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = WorkspaceManager(SimpleMock())

    def test_legacy_plot_arg_detectors(self) -> None:
        self.assertTrue(self.manager._is_legacy_discrete_plot_args({"rectangle_names": ["r1"]}))
        self.assertTrue(self.manager._is_legacy_discrete_plot_args({"fill_area_names": ["a1"]}))
        self.assertFalse(self.manager._is_legacy_discrete_plot_args({}))

        self.assertTrue(self.manager._is_legacy_continuous_plot_args({"function_name": "f"}))
        self.assertTrue(self.manager._is_legacy_continuous_plot_args({"fill_area_name": "fa"}))
        self.assertFalse(self.manager._is_legacy_continuous_plot_args({}))

    def test_is_workspace_management_expression(self) -> None:
        self.assertTrue(self.manager._is_workspace_management_expression("list_workspaces()"))
        self.assertTrue(self.manager._is_workspace_management_expression("save_workspace('w')"))
        self.assertTrue(self.manager._is_workspace_management_expression("load_workspace('w')"))
        self.assertFalse(self.manager._is_workspace_management_expression("delete_workspace('w')"))
        self.assertFalse(self.manager._is_workspace_management_expression("plot_distribution()"))

    def test_response_helper_methods(self) -> None:
        req = SimpleMock(text='{"status":"success","data":{"state":{"Points":[]}}}')
        response = self.manager._response_from_request(req)

        self.assertTrue(self.manager._response_is_success(response))
        self.assertEqual(self.manager._workspace_state_from_response(response), {"Points": []})
        self.assertEqual(self.manager._workspace_list_from_response({"data": ["a", "b"]}), ["a", "b"])
        self.assertEqual(self.manager._workspace_list_from_response({}), [])
        self.assertEqual(
            self.manager._format_workspace_error("loading", {"message": "boom"}),
            "Error loading workspace: boom",
        )


class TestWorkspaceManagerOrchestration(unittest.TestCase):
    def test_execute_sync_request_runs_build_open_finalize_in_order(self) -> None:
        manager = WorkspaceManager(SimpleMock())
        events = []

        def fake_build(on_complete: object, error_prefix: str) -> str:
            events.append(("build", error_prefix))
            return "REQ"

        def fake_open(req: object, method: str, url: str) -> None:
            events.append(("open", req, method, url))

        def fake_finalize(req: object, on_complete: object) -> str:
            events.append(("finalize", req))
            return "ok"

        manager._build_sync_request = fake_build  # type: ignore[assignment]
        manager._open_and_send_sync_request = fake_open  # type: ignore[assignment]
        manager._finalize_sync_request = fake_finalize  # type: ignore[assignment]

        result = manager._execute_sync_request(
            method="GET",
            url="/list_workspaces",
            on_complete=lambda req: "unused",
            error_prefix="Error listing workspaces",
        )

        self.assertEqual(result, "ok")
        self.assertEqual(
            events,
            [
                ("build", "Error listing workspaces"),
                ("open", "REQ", "GET", "/list_workspaces"),
                ("finalize", "REQ"),
            ],
        )

    def test_execute_sync_json_request_runs_build_open_finalize_in_order(self) -> None:
        manager = WorkspaceManager(SimpleMock())
        events = []
        payload = {"state": {"Points": []}, "name": "w1"}

        def fake_build(on_complete: object, error_prefix: str) -> str:
            events.append(("build", error_prefix))
            return "REQ"

        def fake_open(req: object, method: str, url: str, sent_payload: object) -> None:
            events.append(("open_json", req, method, url, sent_payload))

        def fake_finalize(req: object, on_complete: object) -> str:
            events.append(("finalize", req))
            return "ok-json"

        manager._build_sync_request = fake_build  # type: ignore[assignment]
        manager._open_and_send_sync_json_request = fake_open  # type: ignore[assignment]
        manager._finalize_sync_request = fake_finalize  # type: ignore[assignment]

        result = manager._execute_sync_json_request(
            method="POST",
            url="/save_workspace",
            payload=payload,
            on_complete=lambda req: "unused",
            error_prefix="Error saving workspace",
        )

        self.assertEqual(result, "ok-json")
        self.assertEqual(
            events,
            [
                ("build", "Error saving workspace"),
                ("open_json", "REQ", "POST", "/save_workspace", payload),
                ("finalize", "REQ"),
            ],
        )

    def test_save_workspace_uses_json_request_and_drops_bars(self) -> None:
        state = {"Bars": [{"name": "bar1"}], "Points": [{"name": "A"}]}
        canvas = SimpleMock(get_canvas_state=lambda: state)
        manager = WorkspaceManager(canvas)
        captured = {}

        def fake_execute_json(
            method: str,
            url: str,
            payload: dict,
            on_complete: object,
            error_prefix: str,
        ) -> str:
            captured["method"] = method
            captured["url"] = url
            captured["payload"] = payload
            captured["error_prefix"] = error_prefix
            return on_complete(SimpleMock(text='{"status":"success"}'))  # type: ignore[misc]

        manager._execute_sync_json_request = fake_execute_json  # type: ignore[assignment]

        result = manager.save_workspace("ws1")

        self.assertEqual(result, 'Workspace "ws1" saved successfully.')
        self.assertEqual(captured.get("method"), "POST")
        self.assertEqual(captured.get("url"), "/save_workspace")
        self.assertEqual(captured.get("error_prefix"), "Error saving workspace")
        payload = captured["payload"]
        self.assertEqual(payload["name"], "ws1")
        self.assertNotIn("Bars", payload["state"])
        self.assertIn("Points", payload["state"])

    def test_parse_save_workspace_response_handles_invalid_json(self) -> None:
        manager = WorkspaceManager(SimpleMock())
        req = SimpleMock(text="{not-json")
        message = manager._parse_save_workspace_response(req, "bad")
        self.assertTrue(message.startswith("Error saving workspace:"))

    def test_parse_load_workspace_response_success_restores_state(self) -> None:
        manager = WorkspaceManager(SimpleMock())
        restored = {}

        def fake_restore(state: dict) -> None:
            restored["state"] = state

        manager._restore_workspace_state = fake_restore  # type: ignore[assignment]
        req = SimpleMock(text='{"status":"success","data":{"state":{"Points":[{"name":"A"}]}}}')

        message = manager._parse_load_workspace_response(req, "demo")

        self.assertEqual(message, 'Workspace "demo" loaded successfully.')
        self.assertEqual(restored.get("state"), {"Points": [{"name": "A"}]})

    def test_parse_load_workspace_response_missing_state_message(self) -> None:
        manager = WorkspaceManager(SimpleMock())
        req = SimpleMock(text='{"status":"success","data":{}}')

        message = manager._parse_load_workspace_response(req, None)

        self.assertEqual(message, "Error loading workspace: No state data found in response")

    def test_parse_list_workspaces_response_success_formats_csv(self) -> None:
        manager = WorkspaceManager(SimpleMock())
        req = SimpleMock(text='{"status":"success","data":["w1","w2"]}')

        message = manager._parse_list_workspaces_response(req)

        self.assertEqual(message, "w1, w2")

    def test_parse_delete_workspace_response_error_uses_error_contract(self) -> None:
        manager = WorkspaceManager(SimpleMock())
        req = SimpleMock(text='{"status":"error","message":"missing"}')

        message = manager._parse_delete_workspace_response(req, "x")

        self.assertEqual(message, "Error deleting workspace: missing")


if __name__ == "__main__":
    unittest.main()
