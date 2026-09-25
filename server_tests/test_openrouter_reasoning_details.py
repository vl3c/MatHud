"""
Tests for OpenRouter-specific message handling.

OpenRouter returns a model's reasoning as ``reasoning_details`` on the assistant
message (and as fragments in streamed deltas). Gemini 3 models require those
details, which carry their thought signatures, to be sent back unchanged with
the assistant message on the request that follows a tool call. OpenRouter also
gets the system prompt as role "system" for non-OpenAI models.
"""

from __future__ import annotations

import copy
import json
import unittest
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List
from unittest.mock import MagicMock, patch

from static.ai_model import AIModel
from static.openai_completions_api import OpenAIChatCompletionsAPI, accumulate_reasoning_details
from static.providers.openrouter_api import OpenRouterAPI

GEMINI = "google/gemini-3.8-flash"

REASONING_DETAILS: List[Dict[str, Any]] = [
    {"type": "reasoning.text", "text": "Need a circle.", "format": "google-gemini-v1", "index": 0},
    {"type": "reasoning.encrypted", "data": "c2lnbmF0dXJl", "id": "tool_1", "format": "google-gemini-v1", "index": 1},
]


def _make_openrouter(model_id: str = GEMINI) -> tuple[OpenRouterAPI, MagicMock]:
    with patch("static.providers.openrouter_api.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            api = OpenRouterAPI(model=AIModel.from_identifier(model_id), tools=[])
    return api, mock_client


def _record_sent_messages(mock_client: MagicMock, responses: List[Any]) -> List[List[Dict[str, Any]]]:
    """Make the mock return ``responses`` in turn and snapshot the messages of each request."""
    sent: List[List[Dict[str, Any]]] = []
    queue = list(responses)

    def create(**kwargs: Any) -> Any:
        sent.append(copy.deepcopy(kwargs["messages"]))
        return queue.pop(0)

    mock_client.chat.completions.create.side_effect = create
    return sent


def _tool_call(call_id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name="create_circle", arguments='{"r": 1}'))


def _completion(message: SimpleNamespace, finish_reason: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], usage=None)


def _user_prompt(text: str) -> str:
    return json.dumps({"user_message": text, "use_vision": False})


def _tool_results_prompt(call_id: str = "call_1") -> str:
    results = [{"tool_call_id": call_id, "result": {"status": "ok"}}]
    return json.dumps({"tool_call_results": json.dumps(results), "use_vision": False})


def _chunk(delta: SimpleNamespace, finish_reason: Any = None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)], usage=None)


def _delta(**fields: Any) -> SimpleNamespace:
    base: Dict[str, Any] = {"content": None, "tool_calls": None, "reasoning_details": None}
    base.update(fields)
    return SimpleNamespace(**base)


def _assistant_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [m for m in messages if m.get("role") == "assistant"]


class TestReasoningDetailsNonStreamed(unittest.TestCase):
    def test_reasoning_details_are_sent_back_after_tool_call(self) -> None:
        api, mock_client = _make_openrouter()
        first = _completion(
            SimpleNamespace(content="", tool_calls=[_tool_call()], reasoning_details=REASONING_DETAILS),
            "tool_calls",
        )
        second = _completion(SimpleNamespace(content="Done.", tool_calls=None), "stop")
        sent = _record_sent_messages(mock_client, [first, second])

        api.create_chat_completion(_user_prompt("draw a circle"))
        api.create_chat_completion(_tool_results_prompt())

        assistant = _assistant_messages(sent[1])
        self.assertEqual(len(assistant), 1)
        self.assertEqual(assistant[0]["reasoning_details"], REASONING_DETAILS)
        self.assertEqual(assistant[0]["tool_calls"][0]["id"], "call_1")

    def test_sdk_objects_are_stored_as_dicts(self) -> None:
        api, _ = _make_openrouter()
        detail = MagicMock()
        detail.model_dump.return_value = dict(REASONING_DETAILS[1])
        message = SimpleNamespace(content="", tool_calls=[_tool_call()], reasoning_details=[detail])

        stored = api._create_assistant_message(message)

        self.assertEqual(stored["reasoning_details"], [REASONING_DETAILS[1]])

    def test_no_reasoning_details_key_when_absent(self) -> None:
        api, _ = _make_openrouter()
        stored = api._create_assistant_message(SimpleNamespace(content="Hi", tool_calls=None))
        self.assertNotIn("reasoning_details", stored)

    def test_history_cleanup_keeps_reasoning_details(self) -> None:
        api, mock_client = _make_openrouter()
        first = _completion(
            SimpleNamespace(content="", tool_calls=[_tool_call()], reasoning_details=REASONING_DETAILS),
            "tool_calls",
        )
        _record_sent_messages(mock_client, [first])
        prompt = json.dumps(
            {
                "user_message": "draw a circle",
                "use_vision": True,
                "canvas_snapshot": "data:image/png;base64,AAAA",
                "canvas_state": {"Points": []},
            }
        )

        api.create_chat_completion(prompt)
        api._clean_conversation_history()

        self.assertEqual(_assistant_messages(api.messages)[0]["reasoning_details"], REASONING_DETAILS)

    def test_reasoning_details_of_earlier_turns_are_not_resent(self) -> None:
        """A new user prompt drops reasoning_details from earlier turns; within the
        current turn's tool loop they are still sent back."""
        api, mock_client = _make_openrouter()
        turn1_details = [dict(REASONING_DETAILS[0], text="Turn one.")]
        responses = [
            _completion(
                SimpleNamespace(content="", tool_calls=[_tool_call("call_1")], reasoning_details=turn1_details),
                "tool_calls",
            ),
            _completion(SimpleNamespace(content="Done.", tool_calls=None, reasoning_details=turn1_details), "stop"),
            _completion(
                SimpleNamespace(content="", tool_calls=[_tool_call("call_2")], reasoning_details=REASONING_DETAILS),
                "tool_calls",
            ),
            _completion(SimpleNamespace(content="Done again.", tool_calls=None), "stop"),
        ]
        sent = _record_sent_messages(mock_client, responses)

        api.create_chat_completion(_user_prompt("draw a circle"))
        api.create_chat_completion(_tool_results_prompt("call_1"))
        api.create_chat_completion(_user_prompt("draw another circle"))
        api.create_chat_completion(_tool_results_prompt("call_2"))

        turn2_first_request = _assistant_messages(sent[2])
        self.assertEqual(len(turn2_first_request), 2)
        self.assertTrue(all("reasoning_details" not in m for m in turn2_first_request))

        turn2_tool_loop = _assistant_messages(sent[3])
        self.assertEqual(len(turn2_tool_loop), 3)
        self.assertTrue(all("reasoning_details" not in m for m in turn2_tool_loop[:2]))
        self.assertEqual(turn2_tool_loop[2]["reasoning_details"], REASONING_DETAILS)
        self.assertEqual(turn2_tool_loop[2]["tool_calls"][0]["id"], "call_2")

    @patch("static.openai_api_base.OpenAI")
    def test_openai_chat_completions_does_not_keep_reasoning_details(self, _mock_openai: MagicMock) -> None:
        api = OpenAIChatCompletionsAPI()
        message = SimpleNamespace(content="", tool_calls=[_tool_call()], reasoning_details=REASONING_DETAILS)
        self.assertNotIn("reasoning_details", api._create_assistant_message(message))


class TestReasoningDetailsStreamed(unittest.TestCase):
    def _tool_call_stream(self) -> Iterator[Any]:
        yield _chunk(
            _delta(
                reasoning_details=[
                    {"type": "reasoning.text", "text": "Need ", "format": "google-gemini-v1", "index": 0}
                ]
            )
        )
        yield _chunk(_delta(reasoning_details=[{"type": "reasoning.text", "text": "a circle.", "index": 0}]))
        yield _chunk(
            _delta(
                tool_calls=[
                    {"index": 0, "id": "call_1", "function": {"name": "create_circle", "arguments": '{"r": 1}'}}
                ],
                reasoning_details=[dict(REASONING_DETAILS[1])],
            )
        )
        yield _chunk(_delta(), finish_reason="tool_calls")

    def _answer_stream(self) -> Iterator[Any]:
        yield _chunk(_delta(content="Done."))
        yield _chunk(_delta(), finish_reason="stop")

    def test_streamed_reasoning_details_are_merged_and_sent_back(self) -> None:
        api, mock_client = _make_openrouter()
        sent = _record_sent_messages(mock_client, [self._tool_call_stream(), self._answer_stream()])

        list(api.create_chat_completion_stream(_user_prompt("draw a circle")))
        list(api.create_chat_completion_stream(_tool_results_prompt()))

        assistant = _assistant_messages(sent[1])
        self.assertEqual(len(assistant), 1)
        self.assertEqual(assistant[0]["reasoning_details"], REASONING_DETAILS)
        self.assertEqual(assistant[0]["tool_calls"][0]["id"], "call_1")
        self.assertNotIn("reasoning_details", _assistant_messages(api.messages)[-1])

    def test_accumulate_keeps_separate_indices_and_types(self) -> None:
        accumulator: List[Dict[str, Any]] = []
        accumulate_reasoning_details([{"type": "reasoning.text", "text": "a", "index": 0}], accumulator)
        accumulate_reasoning_details([{"type": "reasoning.summary", "summary": "s", "index": 0}], accumulator)
        accumulate_reasoning_details(
            [{"type": "reasoning.text", "text": "b", "index": 0, "signature": None}], accumulator
        )
        accumulate_reasoning_details([{"type": "reasoning.text", "signature": "sig", "index": 0}], accumulator)
        accumulate_reasoning_details(None, accumulator)

        self.assertEqual(
            accumulator,
            [
                {"type": "reasoning.text", "text": "ab", "index": 0, "signature": "sig"},
                {"type": "reasoning.summary", "summary": "s", "index": 0},
            ],
        )

    def test_accumulate_merges_untyped_fragment_by_index(self) -> None:
        """A later fragment without a type continues the entry with its index."""
        accumulator: List[Dict[str, Any]] = []
        accumulate_reasoning_details([{"type": "reasoning.text", "text": "Need ", "index": 0}], accumulator)
        accumulate_reasoning_details([{"type": "reasoning.encrypted", "data": "c2ln", "index": 1}], accumulator)
        accumulate_reasoning_details([{"text": "a circle.", "index": 0}], accumulator)
        accumulate_reasoning_details([{"data": "bmF0dXJl", "index": 1, "type": None}], accumulator)

        self.assertEqual(
            accumulator,
            [
                {"type": "reasoning.text", "text": "Need a circle.", "index": 0},
                {"type": "reasoning.encrypted", "data": "c2lnbmF0dXJl", "index": 1},
            ],
        )


class TestOpenRouterSystemRole(unittest.TestCase):
    def test_non_openai_model_uses_system_role(self) -> None:
        api, _ = _make_openrouter(GEMINI)
        self.assertEqual(api.messages[0]["role"], "system")

    def test_openai_model_keeps_developer_role(self) -> None:
        api, _ = _make_openrouter("openai/gpt-6-luna")
        self.assertEqual(api.messages[0]["role"], "developer")

    def test_role_follows_model_changes_and_reset(self) -> None:
        api, mock_client = _make_openrouter(GEMINI)
        api.set_model("openai/gpt-6-luna")
        self.assertEqual(api.messages[0]["role"], "developer")
        api.set_model("anthropic/claude-sonnet-5")
        self.assertEqual(api.messages[0]["role"], "system")
        api.reset_conversation()
        self.assertEqual(api.messages[0]["role"], "system")

        sent = _record_sent_messages(mock_client, [_completion(SimpleNamespace(content="Hi", tool_calls=None), "stop")])
        api.create_chat_completion(_user_prompt("hi"))
        self.assertEqual(sent[0][0]["role"], "system")

    @patch("static.openai_api_base.OpenAI")
    def test_openai_direct_keeps_developer_role(self, _mock_openai: MagicMock) -> None:
        api = OpenAIChatCompletionsAPI()
        self.assertEqual(api.messages[0]["role"], "developer")


if __name__ == "__main__":
    unittest.main()
