"""Chat text must never be parsed as HTML or turned into script links.

Brython's ``html.SPAN(str)`` parses its string argument as HTML, so user messages,
tool names, arguments and results are set with ``.text``. These tests restore a
chat carrying HTML payloads (as a tampered workspace file could) and render the
same payloads live, then check that no element was created from them. They also
check that MathJax does not load its html extension, whose ``\\href`` would make
``javascript:`` links out of chat math.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from browser import window

from tool_call_log_manager import ToolCallLogManager
from .simple_mock import get_class_attr
from .test_chat_persistence import _DetachedChatUI

IMG_PAYLOAD = '<img src=x onerror="window.__xss=1">'
BOLD_PAYLOAD = "<b>x</b>"
PAYLOADS = (BOLD_PAYLOAD, IMG_PAYLOAD)


def _children_with_class(parent: Any, class_name: str) -> List[Any]:
    return [child for child in parent.children if class_name in get_class_attr(child).split()]


def _xss_flag() -> Any:
    return getattr(window, "__xss", None)


class TestChatHtmlSafety(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = _DetachedChatUI()

    def _assert_literal_text(self, element: Any, expected: str) -> None:
        self.assertEqual(len(element.children), 0, f"markup was parsed: {element.innerHTML}")
        self.assertEqual(element.text, expected)

    def _restored_chat(self, payload: str) -> Dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "text": f"hello {payload}"},
                {
                    "role": "assistant",
                    "text": "Done.",
                    "tools": [{"name": f"tool{payload}", "args": f"x: {payload}", "error": True}],
                },
            ],
            "truncated": 0,
        }

    def test_restored_user_text_and_tool_entries_stay_text(self) -> None:
        for payload in PAYLOADS:
            with self.subTest(payload=payload):
                self.ui.restore_transcript(self._restored_chat(payload))
                user_el, ai_el = _children_with_class(self.ui.history, "chat-message")

                self._assert_literal_text(_children_with_class(user_el, "chat-content")[0], f"hello {payload}")

                log = _children_with_class(ai_el, "tool-call-log-dropdown")[0]
                content = _children_with_class(log, "tool-call-log-content")[0]
                entry = _children_with_class(content, "tool-call-entry")[0]
                self._assert_literal_text(_children_with_class(entry, "tool-call-name")[0], f"tool{payload}")
                self._assert_literal_text(_children_with_class(entry, "tool-call-args")[0], f"(x: {payload})")
        self.assertIsNone(_xss_flag())

    def test_live_user_message_stays_text(self) -> None:
        for payload in PAYLOADS:
            with self.subTest(payload=payload):
                element = self.ui.create_message_element("User", payload)
                self._assert_literal_text(_children_with_class(element, "chat-content")[0], payload)
        self.assertIsNone(_xss_flag())

    def test_live_tool_entry_name_args_error_and_result_stay_text(self) -> None:
        log = ToolCallLogManager()
        for payload in PAYLOADS:
            with self.subTest(payload=payload):
                error_entry = log.create_entry_element(
                    {
                        "name": payload,
                        "args_display": payload,
                        "is_error": True,
                        "error_message": f"Error: {payload}",
                    }
                )
                self._assert_literal_text(_children_with_class(error_entry, "tool-call-name")[0], payload)
                self._assert_literal_text(_children_with_class(error_entry, "tool-call-args")[0], f"({payload})")
                self._assert_literal_text(
                    _children_with_class(error_entry, "tool-call-error-msg")[0], f" → Error: {payload}"
                )

                result_entry = log.create_entry_element({"name": "t", "is_error": False, "result_display": payload})
                self._assert_literal_text(_children_with_class(result_entry, "tool-call-result")[0], f" → {payload}")
        self.assertIsNone(_xss_flag())


class _AutocompleteInput:
    value = "/load "


class _AutocompleteHandler:
    def get_commands_list(self) -> List[Any]:
        return []


class TestAutocompleteHtmlSafety(unittest.TestCase):
    """Autocomplete entries (workspace names, model ids) are shown as text."""

    def test_entry_name_and_description_stay_text(self) -> None:
        from browser import html
        from command_autocomplete import CommandAutocomplete

        autocomplete = CommandAutocomplete(_AutocompleteInput(), _AutocompleteHandler())  # type: ignore[arg-type]
        autocomplete.popup_element = html.DIV()
        for payload in PAYLOADS:
            with self.subTest(payload=payload):
                autocomplete.filtered_commands = [(f"/load {payload}", f"Load workspace '{payload}'")]
                autocomplete._render_filtered_commands()
                item = _children_with_class(autocomplete.popup_element, "command-autocomplete-item")[0]
                name = _children_with_class(item, "command-name")[0]
                desc = _children_with_class(item, "command-description")[0]
                self.assertEqual(len(name.children), 0, name.innerHTML)
                self.assertEqual(name.text, f"/load {payload}")
                self.assertEqual(len(desc.children), 0, desc.innerHTML)
                self.assertEqual(desc.text, f"Load workspace '{payload}'")
        self.assertIsNone(_xss_flag())


class TestMathJaxHrefDisabled(unittest.TestCase):
    """Chat math must not become links, overlays or styled/identified elements.

    Covers ``\\href`` (html extension off, ``\\require`` removed) and what base TeX can
    still set, ``\\mmlToken`` attributes and ``\\bbox`` styles, which the ``ui/safe``
    component filters. Ordinary math must still render.

    The harness is synchronous, so these tests use ``tex2chtml``. For TeX that needs an
    autoloaded extension that is not loaded yet, a synchronous call throws "MathJax
    retry", and a second call while that load is pending can break the extension for
    the rest of the page (a TypeError, or the command left undefined). So these tests
    never typeset extension-backed TeX before the extension's component is loaded:
    ``_require_extension`` checks ``MathJax._.input.tex.<name>`` without typesetting and
    skips the case until then. ``_preload_extensions`` starts loading the extensions
    asynchronously when this module is imported, so they are normally ready by the time
    the class runs. Math that needs no extension is always typeset strictly.
    """

    def _mathjax(self) -> Any:
        mathjax = getattr(window, "MathJax", None)
        if mathjax is None or not hasattr(mathjax, "tex2chtml"):
            self.skipTest("MathJax is not ready")
        return mathjax

    def _require_extension(self, name: str) -> Any:
        """Return MathJax once TeX extension ``name`` is loaded; skip (without typesetting) before."""
        mathjax = self._mathjax()
        if not _extension_loaded(mathjax, name):
            self.skipTest(f"MathJax TeX extension {name!r} is not loaded yet")
        return mathjax

    def _assert_inert(self, node: Any) -> None:
        """No link, no fixed positioning, no external url(), no id or class ``evil``."""
        self.assertIsNone(node.querySelector("a"))
        self.assertIsNone(node.querySelector("[href]"))
        self.assertIsNone(node.querySelector("#evil"))
        self.assertIsNone(node.querySelector(".evil"))
        for element in [node, *node.querySelectorAll("[style]")]:
            style = str(element.getAttribute("style") or "")
            self.assertNotIn("fixed", style)
            self.assertNotIn("url(", style)

    def _assert_renders(self, node: Any) -> None:
        self.assertIsNone(node.querySelector("mjx-merror"))
        self.assertIsNone(node.querySelector("[data-mjx-error]"))

    def test_safe_extension_is_loaded(self) -> None:
        self.assertTrue(bool(self._mathjax()._.ui.safe), "ui/safe is not loaded")

    def test_mmltoken_href_is_removed(self) -> None:
        node = self._mathjax().tex2chtml("\\mmlToken{mi}[href=javascript:window.__xss=(window.__xss||0)+1]{x}")
        self._assert_inert(node)

    def test_mmltoken_full_page_link_overlay_is_removed(self) -> None:
        node = self._mathjax().tex2chtml(
            "\\mmlToken{mtext}[href=javascript:window.__xss=(window.__xss||0)+1,"
            "style='position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:99999;opacity:0']{x}"
        )
        self._assert_inert(node)

    def test_mmltoken_id_and_class_are_removed(self) -> None:
        self._assert_inert(self._mathjax().tex2chtml("\\mmlToken{mi}[id=evil,class=evil]{x}"))

    def test_bbox_cannot_position_or_load_urls(self) -> None:
        node = self._require_extension("bbox").tex2chtml(
            "\\bbox[position:fixed;top:0;left:0;background:url(//example.invalid/a)]{x}"
        )
        self._assert_inert(node)

    def test_html_extension_is_not_autoloaded(self) -> None:
        autoload = self._mathjax().config.tex.autoload
        self.assertEqual(len(autoload.html), 0)

    def test_href_does_not_create_a_link(self) -> None:
        # Strict: with the html extension off, \href needs no load; a "retry" fails the test.
        node = self._mathjax().tex2chtml("\\href{javascript:window.__xss=1}{\\text{click}}")
        self.assertIsNone(node.querySelector("a"))
        self.assertIsNone(node.querySelector("[href]"))

    def test_require_html_does_not_create_a_link(self) -> None:
        # Strict: \require is removed, so a "retry" here would mean the html extension loads.
        node = self._mathjax().tex2chtml(
            "\\require{html}\\href{javascript:void(window.__xss=(window.__xss||0)+1)}{\\text{click}}"
        )
        self.assertIsNone(node.querySelector("a"))
        self.assertIsNone(node.querySelector("[href]"))

    def test_require_html_does_not_enable_class_style_or_id(self) -> None:
        node = self._mathjax().tex2chtml("\\require{html}\\class{evil}{x}\\cssId{evil}{y}\\style{color:red}{z}")
        self.assertIsNone(node.querySelector(".evil"))
        self.assertIsNone(node.querySelector("#evil"))

    def test_bundled_math_still_renders(self) -> None:
        for tex in BUNDLED_MATH:
            with self.subTest(tex=tex):
                self._assert_renders(self._mathjax().tex2chtml(tex))

    def test_extension_math_still_renders(self) -> None:
        for extension, tex in EXTENSION_MATH:
            with self.subTest(tex=tex):
                self._assert_renders(self._require_extension(extension).tex2chtml(tex))

    def test_bbox_colors_survive_the_safe_filter(self) -> None:
        node = self._require_extension("bbox").tex2chtml("\\bbox[yellow,5px,border:2px solid red]{x}")
        self._assert_renders(node)
        styles = " ".join(str(el.getAttribute("style") or "") for el in node.querySelectorAll("[style]"))
        self.assertIn("yellow", styles)
        self.assertIn("red", styles)


# Math that tex-mml-chtml.js typesets without loading anything.
BUNDLED_MATH = (
    "\\frac{1}{2}",
    "\\sqrt{x^2+1}",
    "\\begin{pmatrix}1 & 2 \\\\ 3 & 4\\end{pmatrix}",
    "\\text{area} = \\pi r^2",
    "\\mathbb{R}",
)
# (extension, TeX) for math whose extension is autoloaded on first use.
EXTENSION_MATH = (
    ("color", "\\color{red}{x} + \\textcolor{blue}{y}"),
    ("boldsymbol", "\\boldsymbol{v}"),
    ("cancel", "\\cancel{x}"),
    ("bbox", "\\bbox[yellow]{x}"),
    ("bbox", "\\bbox[5px]{x}"),
)


def _extension_loaded(mathjax: Any, name: str) -> bool:
    """Whether TeX extension ``name`` has loaded (its component registers under MathJax._.input.tex)."""
    try:
        return bool(getattr(mathjax._.input.tex, name, None))
    except Exception:
        return False


def _preload_extensions() -> None:
    """Start loading the EXTENSION_MATH extensions asynchronously (see the class docstring)."""
    try:
        mathjax = getattr(window, "MathJax", None)
        if mathjax is not None and hasattr(mathjax, "tex2chtmlPromise"):
            mathjax.tex2chtmlPromise(" ".join(tex for _, tex in EXTENSION_MATH))
    except Exception:
        pass


_preload_extensions()
