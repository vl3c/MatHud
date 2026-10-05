"""Tests for dropping the replies of abandoned requests (a stopped turn or a reset conversation).

The browser's stop only aborts its fetch; the request goes on in the server. A
reply that ends in tool calls used to append those calls and their "Awaiting
result..." placeholders to whatever conversation was current when it arrived,
so a new conversation started with the previous one's tool calls. Every client
here is a fake; nothing reaches a provider.
"""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from typing import Any, Callable, Iterator, List, Optional
from unittest.mock import patch

from static.ai_model import AIModel
from static.app_manager import AppManager
from static.openai_completions_api import OpenAIChatCompletionsAPI
from static.providers.local.local_agent_api import LocalAgentAPI


def tool_chunk(name: str = "create_segment", finish: Optional[str] = None) -> Any:
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

    def __init__(self, chunks: List[Any], during: Optional[dict[int, Callable[[], None]]] = None) -> None:
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


class TestLocalProvider(unittest.TestCase):
    def test_reply_after_a_new_conversation_is_dropped(self) -> None:
        api = local_api()
        stream = FakeStream([tool_chunk(), finish_chunk()], {2: api.reset_conversation})
        api.client = FakeClient(stream)  # type: ignore[assignment]
        events = list(api.create_chat_completion_stream(prompt()))
        self.assertEqual(events, [])
        self.assertEqual(roles(api), ["system"])  # no tool calls, no "Awaiting result..." stubs

    def test_stop_abandons_the_running_request(self) -> None:
        api = local_api()
        stream = FakeStream([tool_chunk(), finish_chunk()], {2: lambda: api.add_partial_assistant_message("Partial")})
        api.client = FakeClient(stream)  # type: ignore[assignment]
        self.assertEqual(list(api.create_chat_completion_stream(prompt())), [])
        self.assertEqual(roles(api), ["system", "user", "assistant"])
        self.assertEqual(api.messages[-1]["content"], "Partial")

    def test_abandoned_stream_stops_reading(self) -> None:
        api = local_api()
        chunks = [text_chunk("a"), text_chunk("b"), text_chunk("c"), finish_chunk("stop")]
        stream = FakeStream(chunks, {1: api.reset_conversation})
        api.client = FakeClient(stream)  # type: ignore[assignment]
        events = list(api.create_chat_completion_stream(prompt()))
        self.assertEqual(events, [{"type": "token", "text": "a"}])
        self.assertTrue(stream.closed)
        self.assertEqual(stream.consumed, 2)

    def test_current_request_is_kept(self) -> None:
        api = local_api()
        api.client = FakeClient(FakeStream([tool_chunk(), finish_chunk()]))  # type: ignore[assignment]
        events = list(api.create_chat_completion_stream(prompt()))
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual(roles(api), ["system", "user", "assistant", "tool"])

    def test_non_streaming_reply_is_dropped(self) -> None:
        api = local_api()
        api.client = FakeClient(FakeStream([], {0: api.reset_conversation}))  # type: ignore[assignment]
        choice = api.create_chat_completion(prompt())
        self.assertEqual(roles(api), ["system"])
        self.assertFalse(getattr(choice.message, "tool_calls", None))


class TestChatCompletionsProvider(unittest.TestCase):
    def make(self) -> OpenAIChatCompletionsAPI:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            return OpenAIChatCompletionsAPI(model=AIModel.from_identifier("gpt-4.1"), tools=[])

    def test_reply_after_reset_is_dropped(self) -> None:
        api = self.make()
        api.client = FakeClient(FakeStream([tool_chunk(), finish_chunk()], {2: api.reset_conversation}))  # type: ignore[assignment]
        self.assertEqual(list(api.create_chat_completion_stream(prompt())), [])
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

    def payload(self) -> dict[str, Any]:
        return {"message": json.dumps({"user_message": "x", "use_vision": False, "ai_model": "gpt-4.1"})}

    def test_stream_stops_once_the_conversation_is_reset(self) -> None:
        api = self.app.ai_api

        def events() -> Iterator[dict[str, Any]]:
            yield {"type": "token", "text": "Hel"}
            api.reset_conversation()  # e.g. POST /new_conversation from another request
            yield {"type": "token", "text": "lo"}
            yield {"type": "final", "ai_message": "Hello", "ai_tool_calls": [], "finish_reason": "stop"}

        with patch.object(OpenAIChatCompletionsAPI, "create_chat_completion_stream", return_value=events()):
            response = self.client.post("/send_message_stream", json=self.payload())
            lines = [json.loads(line) for line in response.data.decode().splitlines() if line.strip()]
        self.assertEqual([e.get("text") for e in lines if e["type"] == "token"], ["Hel"])
        self.assertFalse([e for e in lines if e["type"] == "final"])
        self.assertEqual(self.client.get("/api/requests_in_flight").get_json()["data"]["requests_in_flight"], 0)

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

    def test_save_partial_response_abandons_running_requests(self) -> None:
        generations = {name: api.conversation_generation for name, api in (("chat", self.app.ai_api),)}
        self.client.post("/save_partial_response", json={"partial_message": ""})
        self.assertTrue(self.app.ai_api.is_abandoned(generations["chat"]))
        before = self.app.responses_api.conversation_generation
        self.client.post("/new_conversation")
        self.assertTrue(self.app.responses_api.is_abandoned(before))

    def test_requests_in_flight_counts_running_streams(self) -> None:
        seen: List[int] = []

        def events() -> Iterator[dict[str, Any]]:
            seen.append(self.client.get("/api/requests_in_flight").get_json()["data"]["requests_in_flight"])
            yield {"type": "final", "ai_message": "", "ai_tool_calls": [], "finish_reason": "stop"}

        with patch.object(OpenAIChatCompletionsAPI, "create_chat_completion_stream", return_value=events()):
            self.client.post("/send_message_stream", json=self.payload()).get_data()
        self.assertEqual(seen, [1])
        self.assertEqual(self.client.get("/api/requests_in_flight").get_json()["data"]["requests_in_flight"], 0)


if __name__ == "__main__":
    unittest.main()
