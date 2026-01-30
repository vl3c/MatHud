"""AI chat interface commands for the MatHud CLI.

Provides commands for sending messages to the AI assistant.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional

import click
import requests

from cli.config import DEFAULT_HOST, DEFAULT_PORT
from cli.server import ServerManager


def get_api_base(host: str, port: int) -> str:
    """Get the API base URL."""
    return f"http://{host}:{port}"


def check_server(host: str, port: int) -> bool:
    """Check if server is running."""
    manager = ServerManager(host=host, port=port)
    return manager.is_server_running()


def send_message_stream(
    message: str,
    port: int,
    model: Optional[str] = None,
    use_vision: bool = False,
) -> None:
    """Send a message and stream the response.

    Args:
        message: The message to send.
        port: Server port number.
        model: Optional AI model to use.
        use_vision: Whether to include canvas snapshot for vision.
    """
    base_url = get_api_base(DEFAULT_HOST, port)

    # Build message payload
    message_json = {
        "user_message": message,
        "use_vision": use_vision,
    }
    if model:
        message_json["ai_model"] = model

    payload = {
        "message": json.dumps(message_json),
    }

    try:
        with requests.post(
            f"{base_url}/send_message_stream",
            json=payload,
            stream=True,
            timeout=300,  # 5 minute timeout for streaming
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    event = json.loads(line.decode("utf-8"))
                    event_type = event.get("type")

                    if event_type == "token":
                        # Print token without newline for streaming effect
                        sys.stdout.write(event.get("text", ""))
                        sys.stdout.flush()
                    elif event_type == "reasoning":
                        # Print reasoning tokens (for reasoning models)
                        text = event.get("text", "")
                        if text:
                            sys.stdout.write(click.style(text, dim=True))
                            sys.stdout.flush()
                    elif event_type == "final":
                        # End of response
                        sys.stdout.write("\n")
                        sys.stdout.flush()

                        # Check for errors
                        if event.get("finish_reason") == "error":
                            error_msg = event.get("error_details") or event.get("ai_message", "Unknown error")
                            click.echo(click.style(f"\nError: {error_msg}", fg="red"), err=True)
                            raise SystemExit(1)

                        # Show tool calls if any
                        tool_calls = event.get("ai_tool_calls", [])
                        if tool_calls:
                            click.echo(click.style("\nTool calls:", fg="cyan"))
                            for tc in tool_calls:
                                name = tc.get("function_name") or tc.get("function", {}).get("name", "unknown")
                                args = tc.get("arguments", {})
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except json.JSONDecodeError:
                                        pass
                                click.echo(f"  - {name}: {json.dumps(args)}")
                    elif event_type == "log":
                        # Server log entry
                        level = event.get("level", "info")
                        log_msg = event.get("message", "")
                        if level == "error":
                            click.echo(click.style(f"[LOG] {log_msg}", fg="red"), err=True)
                        elif level == "warning":
                            click.echo(click.style(f"[LOG] {log_msg}", fg="yellow"), err=True)

                except json.JSONDecodeError:
                    # Non-JSON line, skip
                    pass

    except requests.RequestException as e:
        click.echo(click.style(f"Request failed: {e}", fg="red"), err=True)
        raise SystemExit(1)


def send_message_sync(
    message: str,
    port: int,
    model: Optional[str] = None,
    use_vision: bool = False,
) -> dict[str, Any]:
    """Send a message and wait for the complete response.

    Args:
        message: The message to send.
        port: Server port number.
        model: Optional AI model to use.
        use_vision: Whether to include canvas snapshot for vision.

    Returns:
        Response data dictionary.
    """
    base_url = get_api_base(DEFAULT_HOST, port)

    # Build message payload
    message_json: dict[str, Any] = {
        "user_message": message,
        "use_vision": use_vision,
    }
    if model:
        message_json["ai_model"] = model

    payload = {
        "message": json.dumps(message_json),
    }

    try:
        response = requests.post(
            f"{base_url}/send_message",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    except requests.RequestException as e:
        return {"status": "error", "message": str(e)}


@click.group()
def chat() -> None:
    """Interact with the AI assistant."""
    pass


@chat.command()
@click.argument("message")
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Server port (default: {DEFAULT_PORT})",
)
@click.option(
    "--model",
    "-m",
    help="AI model to use (e.g., gpt-4o, claude-3-5-sonnet)",
)
@click.option(
    "--vision",
    "-v",
    is_flag=True,
    help="Include canvas snapshot for vision",
)
@click.option(
    "--no-stream",
    is_flag=True,
    help="Wait for complete response instead of streaming",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output response as JSON (implies --no-stream)",
)
def send(
    message: str,
    port: int,
    model: Optional[str],
    vision: bool,
    no_stream: bool,
    as_json: bool,
) -> None:
    """Send a message to the AI assistant.

    MESSAGE is the text to send to the AI.

    Examples:

      mathud chat send "Create a point at (5, 3) named A"

      mathud chat send "Draw a circle with center A and radius 50" --vision

      mathud chat send "What is the derivative of x^2?" --model gpt-4o
    """
    if not check_server(DEFAULT_HOST, port):
        click.echo(click.style(f"Server is not running on port {port}", fg="red"), err=True)
        raise SystemExit(1)

    if as_json:
        no_stream = True

    if no_stream:
        result = send_message_sync(message, port, model=model, use_vision=vision)

        if as_json:
            click.echo(json.dumps(result, indent=2))
        else:
            if result.get("status") == "error":
                click.echo(click.style(f"Error: {result.get('message', 'Unknown error')}", fg="red"), err=True)
                raise SystemExit(1)

            data = result.get("data", {})
            ai_message = data.get("ai_message", "")
            click.echo(ai_message)

            tool_calls = data.get("ai_tool_calls", [])
            if tool_calls:
                click.echo(click.style("\nTool calls:", fg="cyan"))
                for tc in tool_calls:
                    name = tc.get("function_name") or tc.get("function", {}).get("name", "unknown")
                    args = tc.get("arguments", {})
                    click.echo(f"  - {name}: {json.dumps(args)}")
    else:
        send_message_stream(message, port, model=model, use_vision=vision)


@chat.command("new")
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Server port (default: {DEFAULT_PORT})",
)
def new_conversation(port: int) -> None:
    """Start a new conversation (clear chat history)."""
    if not check_server(DEFAULT_HOST, port):
        click.echo(click.style(f"Server is not running on port {port}", fg="red"), err=True)
        raise SystemExit(1)

    try:
        response = requests.post(
            f"{get_api_base(DEFAULT_HOST, port)}/new_conversation",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "error":
            click.echo(click.style(f"Error: {data.get('message', 'Unknown error')}", fg="red"), err=True)
            raise SystemExit(1)

        click.echo(click.style("New conversation started", fg="green"))

    except requests.RequestException as e:
        click.echo(click.style(f"Request failed: {e}", fg="red"), err=True)
        raise SystemExit(1)
