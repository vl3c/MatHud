"""Tests for dropping the replies of abandoned requests (a stopped, timed-out or superseded turn, or a reset conversation).

The browser's stop only aborts its fetch; the request goes on in the server. A
reply that ends in tool calls used to append those calls and their "Awaiting
result..." placeholders to whatever conversation was current when it arrived,
so a new conversation (or the next turn) started with the old turn's tool calls.
Every client here is a fake; nothing reaches a provider.
"""

from __future__ import annotations

import json
import os
import threading
import time
import unittest
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional
from unittest.mock import patch

from static.ai_model import AIModel
from static.app_manager import AppManager
from static.functions_definitions import FUNCTIONS
from static.openai_completions_api import OpenAIChatCompletionsAPI
from static.providers.local.local_agent_api import LocalAgentAPI


def tool_chunk(name: str = "create_segment") -> Any:
    call = SimpleNamespace(index=0, id="call_0", function=SimpleNamespace(name=name, arguments='{"x1": 0}'))
    delta = SimpleNamespace(content=None, tool_calls=[call], reasoning_content=None, role="assistant")
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)], usage=None)


def finish_chunk(reason: str = "tool_calls") -> Any:
    delta = SimpleNamespace(content=None, tool_calls=None, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=reason)], usage=None)


def text_chunk(text: str) -> Any:
    delta = SimpleNamespace(content=text, tool_calls=None, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)], usage=None)


class FakeStream:
    """An SDK stream: yields ``chunks``, running ``during[i]`` before chunk ``i`` (and ``during[len]`` at the end)."""

    def __init__(self, chunks: List[Any], during: Optional[Dict[int, Callable[[], None]]] = None) -> None:
        self.chunks = chunks
        self.during = during or {}
        self.closed = False
        self.consumed = 0

    def __iter__(self) -> Iterator[Any]:
        for index, chunk in enumerate(self.chunks):
            if index in self.during:
                self.during[index]()
            if self.closed:
                return
            self.consumed += 1
            yield chunk
        if len(self.chunks) in self.during:
            self.during[len(self.chunks)]()

    def close(self) -> None:
        self.closed = True


class FakeClient:
    def __init__(self, stream: FakeStream) -> None:
        self.stream = stream
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return self.stream
        message = SimpleNamespace(
            content="",
            tool_calls=[SimpleNamespace(id="call_0", function=SimpleNamespace(name="create_segment", arguments="{}"))],
        )
        if self.stream.during.get(0):
            self.stream.during[0]()
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")], usage=None)


def prompt(text: str = "Draw a segment") -> str:
    return json.dumps({"user_message": text, "use_vision": False})


def local_api() -> LocalAgentAPI:
    return LocalAgentAPI(model=AIModel.from_identifier("local-model"))


def roles(api: Any) -> List[str]:
    return [str(m.get("role")) for m in api.messages]


def abandoned_only(events: List[Dict[str, Any]]) -> bool:
    """True when the stream ended with nothing but an abandoned final event carrying its metrics."""
    finals = [e for e in events if e.get("type") == "final"]
    return (
        len(finals) == 1
        and finals[0]["finish_reason"] == "abandoned"
        and finals[0]["ai_tool_calls"] == []
        and finals[0]["metrics"]["finish_reason"] == "abandoned"
    )


class TestLocalProvider(unittest.TestCase):
    def test_reply_after_a_new_conversation_is_dropped(self) -> None:
        api = local_api()
        stream = FakeStream([tool_chunk(), finish_chunk()], {2: api.reset_conversation})
        api.client = FakeClient(stream)  # type: ignore[assignment]
        events = list(api.create_chat_completion_stream(prompt()))
        self.assertTrue(abandoned_only(events))
        self.assertEqual(roles(api), ["system"])  # no tool calls, no "Awaiting result..." stubs

    def test_stop_abandons_the_running_request(self) -> None:
        api = local_api()
        stream = FakeStream([tool_chunk(), finish_chunk()], {2: lambda: api.add_partial_assistant_message("Partial")})
        api.client = FakeClient(stream)  # type: ignore[assignment]
        self.assertTrue(abandoned_only(list(api.create_chat_completion_stream(prompt()))))
        self.assertEqual(roles(api), ["system", "user", "assistant"])
        self.assertEqual(api.messages[-1]["content"], "Partial")

    def test_abandoned_stream_stops_reading(self) -> None:
        api = local_api()
        chunks = [text_chunk("a"), text_chunk("b"), text_chunk("c"), finish_chunk("stop")]
        stream = FakeStream(chunks, {1: api.reset_conversation})
        api.client = FakeClient(stream)  # type: ignore[assignment]
        events = list(api.create_chat_completion_stream(prompt()))
        self.assertEqual(events[0], {"type": "token", "text": "a"})
        self.assertTrue(abandoned_only(events[1:]))
        self.assertTrue(stream.closed)
        self.assertEqual(stream.consumed, 2)

    def test_closing_the_generator_closes_the_model_stream(self) -> None:
        # The route closes the provider's generator when it sees the abandonment first.
        api = local_api()
        stream = FakeStream([text_chunk("a"), text_chunk("b"), finish_chunk("stop")])
        api.client = FakeClient(stream)  # type: ignore[assignment]
        events = api.create_chat_completion_stream(prompt())
        self.assertEqual(next(events)["type"], "token")
        events.close()
        self.assertTrue(stream.closed)

    def test_current_request_is_kept(self) -> None:
        api = local_api()
        api.client = FakeClient(FakeStream([tool_chunk(), finish_chunk()]))  # type: ignore[assignment]
        events = list(api.create_chat_completion_stream(prompt()))
        self.assertEqual(events[-1]["finish_reason"], "tool_calls")
        self.assertEqual(roles(api), ["system", "user", "assistant", "tool"])

    def test_non_streaming_reply_is_dropped_with_its_metrics(self) -> None:
        api = local_api()
        api.client = FakeClient(FakeStream([], {0: api.reset_conversation}))  # type: ignore[assignment]
        choice = api.create_chat_completion(prompt())
        self.assertEqual(roles(api), ["system"])
        self.assertFalse(getattr(choice.message, "tool_calls", None))
        assert api.last_response_metrics is not None
        self.assertEqual(api.last_response_metrics["finish_reason"], "abandoned")


class TestGenerationScoping(unittest.TestCase):
    def make(self) -> OpenAIChatCompletionsAPI:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            return OpenAIChatCompletionsAPI(model=AIModel.from_identifier("gpt-4.1"), tools=None)

    def test_reply_after_reset_is_dropped(self) -> None:
        api = self.make()
        api.client = FakeClient(FakeStream([tool_chunk(), finish_chunk()], {2: api.reset_conversation}))  # type: ignore[assignment]
        self.assertTrue(abandoned_only(list(api.create_chat_completion_stream(prompt()))))
        self.assertEqual(roles(api), ["developer"])

    def test_generation_counter(self) -> None:
        api = self.make()
        generation = api.conversation_generation
        self.assertFalse(api.is_abandoned(generation))
        api.add_partial_assistant_message("")
        self.assertTrue(api.is_abandoned(generation))
        later = api.conversation_generation
        api.restore_conversation([{"role": "user", "content": "hi"}])
        self.assertTrue(api.is_abandoned(later))
        latest = api.conversation_generation
        api.begin_turn()
        self.assertTrue(api.is_abandoned(latest))

    def test_stop_applies_only_to_its_own_turn(self) -> None:
        api = self.make()
        old_turn = api.turn_token
        api.begin_turn()  # the next message was sent before the stop arrived
        api.messages.append({"role": "user", "content": "second"})
        self.assertFalse(api.abandon_turn(old_turn, "partial of the first turn"))
        self.assertEqual([m["content"] for m in api.messages[1:]], ["second"])
        current = api.turn_token
        self.assertTrue(api.abandon_turn(current, "partial"))
        self.assertNotEqual(api.turn_token, current)
        self.assertEqual(api.messages[-1]["content"], "partial")

    def test_turn_tokens_differ_between_providers(self) -> None:
        self.assertNotEqual(self.make().turn_token.split(":")[0], self.make().turn_token.split(":")[0])

    def test_abandoning_unloads_searched_tools(self) -> None:
        api = self.make()
        api.set_tool_mode("search")
        api.inject_tools([FUNCTIONS[0]])
        self.assertTrue(api.has_injected_tools())
        api.abandon_requests_in_flight()
        self.assertFalse(api.has_injected_tools())

    def test_a_stop_waits_for_a_reply_being_stored(self) -> None:
        api = self.make()
        generation = api.conversation_generation
        stopped = threading.Event()

        def stop() -> None:
            api.add_partial_assistant_message("")
            stopped.set()

        with api._reply_guard(generation) as current:
            self.assertTrue(current)
            worker = threading.Thread(target=stop)
            worker.start()
            time.sleep(0.1)
            self.assertFalse(stopped.is_set())  # the stop waits until the reply is stored
            self.assertFalse(api.is_abandoned(generation))
        worker.join(5)
        self.assertTrue(stopped.is_set() and api.is_abandoned(generation))


class TestRoutes(unittest.TestCase):
    def setUp(self) -> None:
        self.original = os.environ.get("REQUIRE_AUTH")
        os.environ["REQUIRE_AUTH"] = "false"
        self.app = AppManager.create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        if self.original is None:
            os.environ.pop("REQUIRE_AUTH", None)
        else:
            os.environ["REQUIRE_AUTH"] = self.original

    def payload(self, text: str = "x", tool_results: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"user_message": text, "use_vision": False, "ai_model": "gpt-4.1"}
        if tool_results is not None:
            body["tool_call_results"] = tool_results
            body["user_message"] = None
        return {"message": json.dumps(body)}

    def events(self, response: Any) -> List[Dict[str, Any]]:
        return [json.loads(line) for line in response.data.decode().splitlines() if line.strip()]

    def stream(self, events: Iterator[Dict[str, Any]], **payload: Any) -> List[Dict[str, Any]]:
        with patch.object(OpenAIChatCompletionsAPI, "create_chat_completion_stream", return_value=events):
            return self.events(self.client.post("/send_message_stream", json=self.payload(**payload)))

    def in_flight(self) -> int:
        return int(self.client.get("/api/requests_in_flight").get_json()["data"]["requests_in_flight"])

    def test_stream_starts_with_the_turn(self) -> None:
        def events() -> Iterator[Dict[str, Any]]:
            yield {"type": "final", "ai_message": "hi", "ai_tool_calls": [], "finish_reason": "stop"}

        lines = self.stream(events())
        self.assertEqual(lines[0], {"type": "turn", "turn": self.app.ai_api.turn_token})

    def test_stream_ends_with_an_abandoned_final_once_the_conversation_is_reset(self) -> None:
        api = self.app.ai_api

        def events() -> Iterator[Dict[str, Any]]:
            yield {"type": "token", "text": "Hel"}
            api.reset_conversation()  # e.g. POST /new_conversation from another request
            yield {"type": "token", "text": "lo"}
            yield {"type": "final", "ai_message": "Hello", "ai_tool_calls": [], "finish_reason": "stop"}

        lines = self.stream(events())
        self.assertEqual([e.get("text") for e in lines if e["type"] == "token"], ["Hel"])
        self.assertEqual([e for e in lines if e["type"] == "final"][-1]["finish_reason"], "abandoned")
        self.assertEqual(self.in_flight(), 0)

    def test_provider_that_stops_early_still_ends_the_stream(self) -> None:
        api = self.app.ai_api

        def events() -> Iterator[Dict[str, Any]]:
            yield {"type": "token", "text": "a"}
            api.add_partial_assistant_message("")  # stopped from another tab; the provider just returns

        lines = self.stream(events())
        self.assertEqual(lines[-1]["finish_reason"], "abandoned")

    def test_provider_abandoned_final_is_passed_on_without_tool_handling(self) -> None:
        def events() -> Iterator[Dict[str, Any]]:
            yield {"type": "final", "ai_message": "", "ai_tool_calls": [], "finish_reason": "abandoned", "metrics": {}}

        with patch("static.routes._intercept_search_tools") as intercept:
            lines = self.stream(events())
        intercept.assert_not_called()
        self.assertEqual(lines[-1]["finish_reason"], "abandoned")

    def test_error_sources(self) -> None:
        def provider_error() -> Iterator[Dict[str, Any]]:
            yield {"type": "final", "ai_message": "x", "ai_tool_calls": [], "finish_reason": "error"}

        self.assertEqual(self.stream(provider_error())[-1]["error_source"], "provider")

        def crash() -> Iterator[Dict[str, Any]]:
            raise RuntimeError("bug in the server")
            yield {}

        self.assertEqual(self.stream(crash())[-1]["error_source"], "server")

    def test_late_stop_does_not_drop_the_next_turn(self) -> None:
        api = self.app.ai_api

        def first() -> Iterator[Dict[str, Any]]:
            yield {"type": "token", "text": "slow"}

        old_turn = self.stream(first())[0]["turn"]
        second_turn: List[str] = []

        def second() -> Iterator[Dict[str, Any]]:
            second_turn.append(api.turn_token)
            # The first turn's stop arrives while the second turn's request runs.
            reply = self.client.post("/save_partial_response", json={"partial_message": "slow", "turn": old_turn})
            second_turn.append(reply.get_json()["message"])
            yield {"type": "final", "ai_message": "done", "ai_tool_calls": [], "finish_reason": "stop"}

        lines = self.stream(second(), text="second")
        self.assertEqual(lines[-1]["finish_reason"], "stop")
        self.assertIn("already ended", second_turn[1])
        self.assertFalse(any(m.get("content") == "slow" for m in api.messages))

    def test_stop_with_its_turn_abandons_it(self) -> None:
        api = self.app.ai_api
        turn = api.turn_token
        generation = api.conversation_generation
        reply = self.client.post("/save_partial_response", json={"partial_message": "part", "turn": turn})
        self.assertEqual(reply.get_json()["message"], "Partial response saved.")
        self.assertTrue(api.is_abandoned(generation))
        self.assertEqual(api.messages[-1]["content"], "part")

    def test_stop_and_new_turn_unload_searched_tools(self) -> None:
        api = self.app.ai_api
        api.set_tool_mode("search")
        api.inject_tools([FUNCTIONS[0]])
        self.client.post("/save_partial_response", json={"partial_message": "", "turn": api.turn_token})
        self.assertFalse(api.has_injected_tools())
        api.inject_tools([FUNCTIONS[0]])

        def events() -> Iterator[Dict[str, Any]]:
            yield {"type": "final", "ai_message": "", "ai_tool_calls": [], "finish_reason": "tool_calls"}

        self.stream(events())  # a new user message
        self.assertFalse(api.has_injected_tools())

    def test_tool_results_continue_the_same_turn(self) -> None:
        api = self.app.ai_api
        turn = api.turn_token

        def events() -> Iterator[Dict[str, Any]]:
            yield {"type": "final", "ai_message": "", "ai_tool_calls": [], "finish_reason": "stop"}

        lines = self.stream(events(), tool_results=json.dumps([]))
        self.assertEqual(lines[0]["turn"], turn)

    def test_save_partial_without_a_turn_keeps_the_old_behaviour(self) -> None:
        generation = self.app.ai_api.conversation_generation
        self.client.post("/save_partial_response", json={"partial_message": ""})
        self.assertTrue(self.app.ai_api.is_abandoned(generation))
        before = self.app.responses_api.conversation_generation
        self.client.post("/new_conversation")
        self.assertTrue(self.app.responses_api.is_abandoned(before))

    def test_non_streaming_reply_of_a_stopped_turn_is_not_used(self) -> None:
        api = self.app.ai_api

        def reply(_message: str) -> Any:
            api.add_partial_assistant_message("")  # the user stopped the turn meanwhile
            message = SimpleNamespace(content="", tool_calls=[])
            return SimpleNamespace(message=message, finish_reason="tool_calls")

        with patch.object(OpenAIChatCompletionsAPI, "create_chat_completion", side_effect=reply):
            data = self.client.post("/send_message", json=self.payload()).get_json()["data"]
        self.assertEqual(data["finish_reason"], "abandoned")
        self.assertEqual(data["ai_tool_calls"], [])

    def test_non_streaming_reply_carries_the_turn(self) -> None:
        message = SimpleNamespace(content="hi", tool_calls=[])
        choice = SimpleNamespace(message=message, finish_reason="stop")
        with patch.object(OpenAIChatCompletionsAPI, "create_chat_completion", return_value=choice):
            data = self.client.post("/send_message", json=self.payload()).get_json()["data"]
        self.assertEqual(data["turn"], self.app.ai_api.turn_token)

    def test_requests_in_flight_counts_running_streams(self) -> None:
        seen: List[int] = []

        def events() -> Iterator[Dict[str, Any]]:
            seen.append(self.in_flight())
            yield {"type": "final", "ai_message": "", "ai_tool_calls": [], "finish_reason": "stop"}

        self.stream(events())
        self.assertEqual(seen, [1])
        self.assertEqual(self.in_flight(), 0)


if __name__ == "__main__":
    unittest.main()
