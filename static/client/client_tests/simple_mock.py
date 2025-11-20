from __future__ import annotations

from typing import Any, Dict, List, Tuple


class SimpleMock:
    _attributes: Dict[str, Any]
    _return_value: Any
    calls: List[Tuple[Tuple[Any, ...], Dict[str, Any]]]

    """Very small mock object used by the Brython-side tests."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the mock with optional attributes and return value."""
        object.__setattr__(self, "_attributes", {})
        object.__setattr__(self, "_return_value", None)
        object.__setattr__(
            self,
            "calls",
            [],
        )
        for key, value in kwargs.items():
            if key == "return_value":
                self._return_value = value
            else:
                self._attributes[key] = value

    def __setattr__(self, key: str, value: Any) -> None:
        """Support dynamic attribute assignment while tracking configured values."""
        if key in {"_attributes", "_return_value", "calls"}:
            object.__setattr__(self, key, value)
        else:
            self._attributes[key] = value

    def __getitem__(self, item: str) -> Any:
        """Allow dictionary-style access to configured attributes."""
        if item in self._attributes:
            return self._attributes[item]
        raise KeyError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment for attributes."""
        self._attributes[key] = value

    def __contains__(self, item: object) -> bool:
        """Support ``in`` checks for configured attributes."""
        return item in self._attributes

    def __getattr__(self, attr: str) -> Any:
        """Handle attribute lookup, including ``return_value``."""
        if attr in self._attributes:
            return self._attributes[attr]
        if attr == "return_value":
            return self._return_value
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def setAttribute(self, key: str, value: Any) -> None:
        """Simulate DOM ``setAttribute`` by recording on the attrs dict if present."""
        if "attrs" not in self._attributes or not isinstance(self._attributes["attrs"], dict):
            self._attributes["attrs"] = {}
        self._attributes["attrs"][key] = value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record a call and return the configured ``return_value``."""
        self.calls.append((args, kwargs))
        return self._return_value

    def assert_called_once_with(self, *args: Any, **kwargs: Any) -> None:
        """Assert the mock was called exactly once with the provided arguments."""
        if len(self.calls) != 1:
            raise AssertionError(f"Expected one call, got {len(self.calls)}")
        call_args, call_kwargs = self.calls[0]
        if call_args != args or call_kwargs != kwargs:
            raise AssertionError(
                f"Expected call with ({args}, {kwargs}), got ({call_args}, {call_kwargs})"
            )

    def assert_called_once(self) -> None:
        """Assert the mock was called exactly once (any arguments)."""
        if len(self.calls) != 1:
            raise AssertionError(f"Expected one call, got {len(self.calls)}")

    def assert_called(self) -> None:
        """Assert the mock was called at least once (any arguments)."""
        if not self.calls:
            raise AssertionError("Expected the mock to be called at least once, got 0 calls")

    def assert_not_called(self) -> None:
        """Assert the mock was never invoked."""
        if self.calls:
            raise AssertionError(f"Expected 0 calls, got {len(self.calls)}")

    def reset_mock(self) -> None:
        """Clear recorded calls."""
        object.__setattr__(self, "calls", [])