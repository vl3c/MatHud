"""Chat text must never be parsed as HTML or turned into script links.

Brython's ``html.SPAN(str)`` parses its string argument as HTML, so user messages,
tool names, arguments and results are set with ``.text``. These tests restore a
chat carrying HTML payloads (as a tampered workspace file could) and render the
same payloads live, then check that no element was created from them.

Chat math is untrusted TeX. TestMathJaxHrefDisabled checks the MathJax configuration
(no links, overlays, url() requests or foreign ids/classes from any TeX command), and
TestMathOutputSanitizer checks the output sanitiser that runs after every chat
typeset (static/math_output_sanitizer.js) on hand-built DOM.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from browser import document, window

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


XSS_JS = "window.__xss=(window.__xss||0)+1"
OVERLAY = "position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:99999;opacity:0"
# \mmlToken href values that plain ui/safe let through (control characters inside the
# scheme are not recognised as javascript:) or that are simply unwanted in chat math.
HREF_PAYLOADS = {
    "tab inside scheme": f"java\tscript:{XSS_JS}",
    "tab before colon": f"javascript\t:{XSS_JS}",
    "control character prefix": f"\x01javascript:{XSS_JS}",
    "https": "https://example.invalid/p",
    "protocol-relative": "//example.invalid/p",
    "file": "file:///C:/Windows/win.ini",
}
# MathJax cases skipped in this run (MathJax missing, or an extension not loaded yet);
# test_zz_no_extension_case_was_skipped fails on any.
_SKIPPED_EXTENSION_CASES: List[str] = []


class TestMathJaxHrefDisabled(unittest.TestCase):
    """Chat math must not become links, overlays, network requests or foreign ids/classes.

    Layer A, the MathJax configuration in templates/index.html, is tested here with the
    raw output of ``tex2chtml`` (no sanitiser): the html extension is off and ``\\require``
    removed, ui/safe allows no URLs and no cursor styles, and fontfamily (also set by
    ``\\unicode``'s font option) is filtered. Layer B, the output sanitiser, is tested
    on its own in TestMathOutputSanitizer.

    The harness is synchronous, so these tests use ``tex2chtml``. A synchronous call
    during a pending extension autoload can break that extension for the rest of the
    page, so extension-backed TeX is typeset only once ``_require_extension`` sees the
    extension's component loaded (``MathJax._.input.tex.<name>``, checked without
    typesetting). The page preloads the extensions these tests use when MathJax starts
    (startup.ready in templates/index.html), so in the suite none should be skipped;
    ``test_zz_no_extension_case_was_skipped`` runs last and fails if any was, or if
    MathJax itself is missing (so a MathJax load failure cannot pass as skips).
    """

    def _mathjax(self) -> Any:
        mathjax = _ready_mathjax()
        if mathjax is None:
            _SKIPPED_EXTENSION_CASES.append(f"{self.id()} (MathJax not ready)")
            self.skipTest("MathJax is not ready")
        return mathjax

    def _require_extension(self, name: str) -> Any:
        """Return MathJax once TeX extension ``name`` is loaded; skip (without typesetting) before."""
        mathjax = self._mathjax()
        if not _extension_loaded(mathjax, name):
            _SKIPPED_EXTENSION_CASES.append(f"{self.id()} ({name})")
            print(f"[TestMathJaxHrefDisabled] skipped: MathJax extension {name!r} not loaded yet")
            self.skipTest(f"MathJax TeX extension {name!r} is not loaded yet")
        return mathjax

    def _assert_inert(self, node: Any) -> None:
        """No link, no fixed positioning, no url(), no cursor, no id or class ``evil``."""
        self.assertIsNone(node.querySelector("a"))
        self.assertIsNone(node.querySelector("[href]"))
        self.assertIsNone(node.querySelector("#evil"))
        self.assertIsNone(node.querySelector(".evil"))
        for element in [node, *node.querySelectorAll("[style]")]:
            style = str(element.getAttribute("style") or "")
            self.assertNotIn("fixed", style)
            self.assertNotIn("url(", style)
            self.assertNotIn("cursor", style)

    def _assert_renders(self, node: Any) -> None:
        self.assertIsNone(node.querySelector("mjx-merror"))
        self.assertIsNone(node.querySelector("[data-mjx-error]"))

    # ── Configuration ───────────────────────────────────────────

    def test_safe_extension_is_loaded(self) -> None:
        self.assertTrue(bool(self._mathjax()._.ui.safe), "ui/safe is not loaded")

    def test_safe_options_block_urls_and_cursor(self) -> None:
        options = self._mathjax().startup.document.safe.options
        self.assertEqual(options.allow.URLs, "none")
        self.assertFalse(bool(options.safeStyles.cursor))

    def test_html_extension_is_not_autoloaded(self) -> None:
        autoload = self._mathjax().config.tex.autoload
        self.assertEqual(len(autoload.html), 0)

    # ── Links ───────────────────────────────────────────────────

    def test_href_does_not_create_a_link(self) -> None:
        # Strict: with the html extension off, \href needs no load; a "retry" fails the test.
        node = self._mathjax().tex2chtml(f"\\href{{javascript:{XSS_JS}}}{{\\text{{click}}}}")
        self._assert_inert(node)

    def test_require_html_does_not_create_a_link(self) -> None:
        # Strict: \require is removed, so a "retry" here would mean the html extension loads.
        node = self._mathjax().tex2chtml(f"\\require{{html}}\\href{{javascript:{XSS_JS}}}{{\\text{{click}}}}")
        self._assert_inert(node)

    def test_require_html_does_not_enable_class_style_or_id(self) -> None:
        node = self._mathjax().tex2chtml("\\require{html}\\class{evil}{x}\\cssId{evil}{y}\\style{color:red}{z}")
        self._assert_inert(node)

    def test_mmltoken_hrefs_are_removed(self) -> None:
        for label, href in HREF_PAYLOADS.items():
            with self.subTest(href=label):
                self._assert_inert(self._mathjax().tex2chtml(f"\\mmlToken{{mi}}[href={href}]{{x}}"))

    # ── Overlays, styles and requests ───────────────────────────

    def test_mmltoken_full_page_link_overlay_is_removed(self) -> None:
        node = self._mathjax().tex2chtml(f"\\mmlToken{{mtext}}[href=javascript:{XSS_JS},style='{OVERLAY}']{{x}}")
        self._assert_inert(node)

    def test_fontfamily_overlay_is_removed(self) -> None:
        node = self._mathjax().tex2chtml(f"\\mmlToken{{mi}}[fontfamily='x;{OVERLAY}']{{x}}")
        self._assert_inert(node)

    def test_fontfamily_overlay_with_tab_href_is_removed(self) -> None:
        node = self._mathjax().tex2chtml(f"\\mmlToken{{mi}}[href=java\tscript:{XSS_JS},fontfamily='x;{OVERLAY}']{{x}}")
        self._assert_inert(node)

    def test_unicode_font_overlay_is_removed(self) -> None:
        node = self._require_extension("unicode").tex2chtml(
            f"\\unicode[x;{OVERLAY};background:url(//example.invalid/u)]{{x41}}"
        )
        self._assert_inert(node)

    def test_cursor_url_is_removed(self) -> None:
        node = self._mathjax().tex2chtml("\\mmlToken{mi}[style='cursor:url(https://example.invalid/c.png),auto']{x}")
        self._assert_inert(node)

    def test_mathcolor_and_mathbackground_cannot_inject_styles(self) -> None:
        for attribute in ("mathcolor", "mathbackground"):
            with self.subTest(attribute=attribute):
                node = self._mathjax().tex2chtml(
                    f"\\mmlToken{{mi}}[{attribute}='red;{OVERLAY};background:url(//example.invalid/m)']{{x}}"
                )
                self._assert_inert(node)

    def test_mmltoken_id_and_class_are_removed(self) -> None:
        self._assert_inert(self._mathjax().tex2chtml("\\mmlToken{mi}[id=evil,class=evil]{x}"))

    def test_bbox_cannot_position_or_load_urls(self) -> None:
        node = self._require_extension("bbox").tex2chtml(
            "\\bbox[position:fixed;top:0;left:0;background:url(//example.invalid/a)]{x}"
        )
        self._assert_inert(node)

    # ── Ordinary math ───────────────────────────────────────────

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

    def test_unicode_font_name_is_kept(self) -> None:
        node = self._require_extension("unicode").tex2chtml("\\unicode[Arial]{x41}")
        self._assert_renders(node)
        styles = " ".join(str(el.getAttribute("style") or "") for el in node.querySelectorAll("[style]"))
        self.assertIn("Arial", styles)

    # ── User macros and size limits ─────────────────────────────

    def test_user_macros_cannot_amplify_output(self) -> None:
        # \def-nested macros once expanded this to 90k nodes and ~2 s of blocking.
        amplified = "\\def\\a{" + "x" * 100 + "}\\def\\b{" + "\\a" * 30 + "}\\def\\c{" + "\\b" * 30 + "}\\c"
        started = window.performance.now()
        node = self._mathjax().tex2chtml(" ".join([amplified] * 8))
        elapsed_ms = window.performance.now() - started
        self.assertLess(len(node.querySelectorAll("*")), 20000)
        self.assertLess(elapsed_ms, 1500)

    def test_declare_math_operator_cannot_amplify_output(self) -> None:
        # Each \DeclareMathOperator level is parsed by a fresh sub-parser, so maxMacros
        # never tripped: six levels in ~320 characters once gave 1.2M nodes and 7 s.
        tex = "\\DeclareMathOperator{\\a}{" + "x" * 10 + "}"
        previous = "a"
        for name in "bcdef":
            tex += "\\DeclareMathOperator{\\" + name + "}{" + ("\\" + previous + " ") * 10 + "}"
            previous = name
        started = window.performance.now()
        node = self._mathjax().tex2chtml(tex + "\\" + previous)
        elapsed_ms = window.performance.now() - started
        self.assertLess(len(node.querySelectorAll("*")), 20000)
        self.assertLess(elapsed_ms, 1500)

    def test_user_macro_commands_are_undefined(self) -> None:
        mathjax = self._mathjax()
        # Undefined commands render as their own name (noundefined); environments as an error.
        for tex, shown in (
            ("\\def\\zz{D}\\zz", "\\def"),
            ("\\newcommand{\\zz}{D}\\zz", "\\newcommand"),
            ("\\renewcommand{\\pi}{D}\\pi", "\\renewcommand"),
            ("\\let\\pi=D\\pi", "\\let"),
            ("\\newenvironment{zz}{D}{}\\begin{zz}x\\end{zz}", "Unknown environment"),
            ("\\DeclareMathOperator{\\zz}{D}\\zz", "\\DeclareMathOperator"),
            ("\\DeclareMathOperator*{\\zz}{D}\\zz", "\\DeclareMathOperator"),
        ):
            with self.subTest(tex=tex):
                self.assertIn(shown, str(mathjax.tex2chtml(tex).textContent))
        # Removed from the command maps themselves, so an extension that depends on
        # newcommand (e.g. action) cannot bring them back.
        maps = mathjax._.input.tex.MapHandler.MapHandler
        self.assertEqual(int(maps.getMap("Newcommand-macros").map.size), 0)
        self.assertFalse(bool(maps.getMap("AMSmath-macros").map.has("DeclareMathOperator")))

    def test_definecolor_is_undefined(self) -> None:
        mathjax = self._require_extension("color")
        self.assertIn("\\definecolor", str(mathjax.tex2chtml("\\definecolor{red}{rgb}{0,1,0}").textContent))
        self.assertFalse(bool(mathjax._.input.tex.MapHandler.MapHandler.getMap("color").map.has("definecolor")))

    def test_macro_redefinition_does_not_persist(self) -> None:
        mathjax = self._mathjax()
        mathjax.tex2chtml("\\def\\pi{3}")
        mathjax.tex2chtml("\\renewcommand{\\pi}{3}")
        mathjax.tex2chtml("\\DeclareMathOperator{\\pi}{BAD}\\pi")
        mathjax.tex2chtml("\\DeclareMathOperator*{\\sin}{cos}")
        mathjax.tex2chtml("\\DeclareMathOperator{\\frac}{F}")
        self.assertEqual(str(mathjax.tex2chtml("\\pi").textContent), "π")
        self.assertIn("sin", str(mathjax.tex2chtml("\\sin x").textContent))
        self.assertEqual(str(mathjax.tex2chtml("\\frac{1}{2}").textContent), "12")

    def test_colour_definitions_do_not_persist(self) -> None:
        mathjax = self._require_extension("color")
        mathjax.tex2chtml("\\definecolor{red}{rgb}{0,1,0}")
        styles = _styles_under(mathjax.tex2chtml("\\color{red}{x}"))
        self.assertIn("color: red", styles)
        self.assertNotIn("0, 255, 0", styles)

    def test_labels_do_not_persist(self) -> None:
        mathjax = self._mathjax()
        mathjax.tex2chtml("\\label{mathud-eq} x")
        self.assertNotIn("multiply defined", str(mathjax.tex2chtml("\\label{mathud-eq} y").textContent))

    def test_long_formula_is_replaced_by_a_note(self) -> None:
        mathjax = self._mathjax()
        self.assertIn("formula too long", str(mathjax.tex2chtml("x+" * 2500).textContent))
        self._assert_renders(mathjax.tex2chtml("x+" * 1000))

    # ── Render action and containment (the chat path) ───────────

    def test_sanitizer_is_a_render_action_and_survives_rerender(self) -> None:
        mathjax = self._mathjax()
        document_ = mathjax.startup.document
        action_ids = [str(action.item.id) for action in document_.renderActions]
        self.assertIn("mathudSanitize", action_ids)
        holder = document.createElement("div")
        holder.innerHTML = "\\(\\raise{50em}{R}\\)"
        document.body.appendChild(holder)
        try:
            mathjax.typeset([holder])  # plain MathJax typeset, without typesetAndSanitize
            self.assertNotIn("-50em", _styles_under(holder))
            document_.rerender()  # what the context menu's renderer/scale/accessibility options do
            self.assertNotIn("-50em", _styles_under(holder))
        finally:
            mathjax.typesetClear([holder])
            holder.remove()

    def test_chat_math_is_painted_only_inside_its_box(self) -> None:
        mathjax = self._mathjax()
        history = document.getElementById("chat-history")
        if history is None:
            self.fail("#chat-history is missing")
        holder = document.createElement("div")
        holder.innerHTML = '<span class="chat-content">\\(\\smash{\\raise{9em}{\\raise{9em}{R}}}\\) $$x$$</span>'
        history.appendChild(holder)
        try:
            mathjax.typeset([holder])
            # Top-level formulas only (not MathJax's hidden assistive copies).
            inline, display = holder.querySelectorAll("span.chat-content > mjx-container")
            inline_style = window.getComputedStyle(inline)
            self.assertIn("paint", str(inline_style.contain))
            self.assertEqual(str(inline_style.display), "inline-block")
            self.assertNotEqual(str(inline_style.overflowClipMargin), "0px", "accents and italics need a clip margin")
            display_style = window.getComputedStyle(display)
            self.assertIn("paint", str(display_style.contain))
            self.assertEqual(str(display_style.overflowX), "auto")
            # The raised glyph lies outside its container's box, where it is not painted.
            glyph = inline.querySelector("mjx-c").getBoundingClientRect()
            self.assertLess(glyph.bottom, inline.getBoundingClientRect().top)
        finally:
            mathjax.typesetClear([holder])
            holder.remove()

    def test_chat_reads_tex_only(self) -> None:
        # The MathML input would re-typeset the hidden assistive MathML of every earlier
        # formula on each chat typeset.
        input_jax = [str(jax.name) for jax in self._mathjax().startup.document.inputJax]
        self.assertEqual(input_jax, ["TeX"])

    def test_typesetting_messages_one_at_a_time_stays_linear(self) -> None:
        """20 messages typeset one by one, as the chat does: no nested copies, bounded time."""
        mathjax = self._mathjax()
        safety = window.MatHudMathSafety
        holder = document.createElement("div")
        document.body.appendChild(holder)
        per_message_ms: List[float] = []
        try:
            for index in range(20):
                message = document.createElement("div")
                message.innerHTML = (
                    f"<p>\\(\\frac{{a_{index}}}{{b}}\\), \\(\\sqrt{{x+{index}}}\\) and \\(\\sum_k k\\)</p>"
                    f"<p>$$\\int_0^{{{index}}} f(x)\\,dx$$</p>"
                )
                holder.appendChild(message)
                started = window.performance.now()
                mathjax.typeset([holder])  # the chat typesets the whole history each time
                safety.sanitize(holder, True)
                per_message_ms.append(window.performance.now() - started)
            self.assertEqual(len(holder.querySelectorAll("mjx-container mjx-container")), 0)
            self.assertEqual(len(holder.querySelectorAll("mjx-container")), 80)
            self.assertLess(max(per_message_ms), 500, per_message_ms)
            self.assertLess(sum(per_message_ms), 5000, per_message_ms)
        finally:
            mathjax.typesetClear([holder])
            holder.remove()

    def test_menu_hides_entries_needing_unvendored_components(self) -> None:
        menu = self._mathjax().startup.document.menu
        self.assertEqual(str(menu.settings.renderer), "CHTML")
        items = list(menu.menu.items)
        for item in list(items):
            submenu = getattr(item, "submenu", None)  # separators have none
            if submenu:
                items.extend(submenu.items)
        hidden = {str(item.id) for item in items if getattr(item, "id", None) and item.isHidden()}
        for entry in ("Renderer", "Language", "Activate", "Collapsible", "AutoCollapse"):
            self.assertIn(entry, hidden)

    def test_zz_no_extension_case_was_skipped(self) -> None:
        """Runs last (alphabetical order): no MathJax case may be skipped silently."""
        self.assertIsNotNone(_ready_mathjax(), "MathJax is not loaded: window.MathJax.tex2chtml is missing")
        self.assertEqual(_SKIPPED_EXTENSION_CASES, [], "MathJax tests were skipped")


class TestMathOutputSanitizer(unittest.TestCase):
    """Layer B, static/math_output_sanitizer.js, on hand-built DOM (no MathJax needed)."""

    def setUp(self) -> None:
        safety = getattr(window, "MatHudMathSafety", None)
        if safety is None:
            self.fail("static/math_output_sanitizer.js is not loaded")
        self.safety = safety
        self.root = document.createElement("div")

    def _container(self, inner_html: str) -> Any:
        container = document.createElement("mjx-container")
        container.setAttribute("class", "MathJax CtxtMenu_Attached_0")
        container.innerHTML = inner_html
        self.root.appendChild(container)
        return container

    def test_unwraps_links_and_keeps_their_content(self) -> None:
        container = self._container(
            '<mjx-math><a href="javascript:alert(1)"><mjx-mi class="mjx-i">x</mjx-mi></a></mjx-math>'
        )
        self.assertEqual(self.safety.sanitize(self.root), 1)
        self.assertIsNone(container.querySelector("a"))
        self.assertIsNotNone(container.querySelector("mjx-mi"))
        self.assertEqual(container.textContent, "x")

    def test_removes_link_and_event_attributes(self) -> None:
        container = self._container(
            '<mjx-mi href="java\tscript:x" src="//e.invalid/i" onclick="x()" onmouseover="x()">x</mjx-mi>'
            '<svg><use xlink:href="//e.invalid/s#a"></use></svg>'
        )
        self.safety.sanitize(self.root)
        for element in container.querySelectorAll("*"):
            names = [attr.name for attr in element.attributes]
            for name in names:
                self.assertFalse(name.lower().startswith("on"), name)
                self.assertFalse(name.lower().endswith("href"), name)
                self.assertNotEqual(name.lower(), "src")

    def test_keeps_only_mathjax_ids_and_classes(self) -> None:
        container = self._container(
            '<mjx-mi id="evil" class="evil mjx-i TEX-I">x</mjx-mi><mjx-mo id="mjx-eqn-1" class="also-evil">y</mjx-mo>'
        )
        self.safety.sanitize(self.root)
        mi = container.querySelector("mjx-mi")
        mo = container.querySelector("mjx-mo")
        self.assertFalse(mi.hasAttribute("id"))
        self.assertEqual(mi.getAttribute("class"), "mjx-i TEX-I")
        self.assertEqual(mo.getAttribute("id"), "mjx-eqn-1")
        self.assertFalse(mo.hasAttribute("class"))
        self.assertEqual(container.getAttribute("class"), "MathJax CtxtMenu_Attached_0")

    def test_drops_positioning_cursor_and_url_styles_keeps_the_rest(self) -> None:
        container = self._container(
            '<mjx-mi style="position: fixed; top: 0; left: 0; z-index: 99999; cursor: pointer; '
            "background: url(//e.invalid/b); color: red; background-color: yellow; "
            'border: 2px solid red; padding: 5px; font-weight: bold; width: 3em">x</mjx-mi>'
            "<mjx-mo style=\"content: 'x'; list-style-image: url(//e.invalid/l)\">y</mjx-mo>"
        )
        self.safety.sanitize(self.root)
        style = container.querySelector("mjx-mi").style
        for prop in ("position", "top", "left", "z-index", "cursor"):
            self.assertEqual(str(style.getPropertyValue(prop)), "", prop)
        mi_style = str(container.querySelector("mjx-mi").getAttribute("style"))
        self.assertNotIn("url(", mi_style)
        self.assertEqual(str(style.getPropertyValue("color")), "red")
        self.assertEqual(str(style.getPropertyValue("background-color")), "yellow")
        self.assertIn("solid", str(style.getPropertyValue("border")))
        self.assertEqual(str(style.getPropertyValue("padding")), "5px")
        self.assertEqual(str(style.getPropertyValue("font-weight")), "bold")
        self.assertEqual(str(style.getPropertyValue("width")), "3em")
        self.assertFalse(container.querySelector("mjx-mo").hasAttribute("style"))

    def test_drops_every_viewport_and_container_query_unit(self) -> None:
        units = ["vw", "vh", "vi", "vb", "vmin", "vmax", "svw", "svh", "svi", "svb", "lvw", "lvh", "lvi", "lvb"]
        units += ["dvw", "dvh", "dvi", "dvb", "cqw", "cqh", "cqi", "cqb", "cqmin", "cqmax"]
        for unit in units:
            with self.subTest(unit=unit):
                container = self._container(f'<mjx-box style="width: 50{unit}; margin-left: 0.5em">x</mjx-box>')
                self.safety.sanitize(self.root)
                style = container.querySelector("mjx-box").style
                self.assertEqual(str(style.getPropertyValue("width")), "", unit)
                self.assertEqual(str(style.getPropertyValue("margin-left")), "0.5em")

    def test_keeps_mathjax_relative_layout_but_not_large_offsets_or_viewport_sizes(self) -> None:
        container = self._container(
            '<mjx-box style="position: relative; top: -0.2em; left: 0.278em; transform: rotate(-0.7rad)">a</mjx-box>'
            '<mjx-box style="position: relative; top: -5000px; left: -100em">b</mjx-box>'
            '<mjx-box style="position: absolute; top: 0.1em">c</mjx-box>'
            '<mjx-box style="width: 100vw; height: 100vh; margin-left: 0.5em">d</mjx-box>'
        )
        self.safety.sanitize(self.root)
        boxes = container.querySelectorAll("mjx-box")
        kept = boxes[0].style
        self.assertEqual(str(kept.getPropertyValue("position")), "relative")
        self.assertEqual(str(kept.getPropertyValue("top")), "-0.2em")
        self.assertEqual(str(kept.getPropertyValue("left")), "0.278em")
        self.assertIn("rotate", str(kept.getPropertyValue("transform")))
        moved = boxes[1].style
        self.assertEqual(str(moved.getPropertyValue("position")), "relative")
        self.assertEqual(str(moved.getPropertyValue("top")), "")
        self.assertEqual(str(moved.getPropertyValue("left")), "")
        self.assertFalse(boxes[2].hasAttribute("style"), "absolute position and its offset are dropped")
        sized = boxes[3].style
        self.assertEqual(str(sized.getPropertyValue("width")), "")
        self.assertEqual(str(sized.getPropertyValue("height")), "")
        self.assertEqual(str(sized.getPropertyValue("margin-left")), "0.5em")

    def test_leaves_content_outside_math_alone(self) -> None:
        outside = document.createElement("span")
        outside.setAttribute("id", "user-note")
        outside.setAttribute("style", "position: relative")
        self.root.appendChild(outside)
        self.assertEqual(self.safety.sanitize(self.root), 0)
        self.assertEqual(outside.getAttribute("id"), "user-note")
        self.assertEqual(outside.getAttribute("style"), "position: relative")

    def test_is_a_no_op_on_legitimate_math(self) -> None:
        mathjax = getattr(window, "MathJax", None)
        if mathjax is None or not hasattr(mathjax, "tex2chtml"):
            self.skipTest("MathJax is not ready")
        cases = list(BUNDLED_MATH) + [tex for name, tex in EXTENSION_MATH if _extension_loaded(mathjax, name)]
        for tex in cases:
            with self.subTest(tex=tex):
                holder = document.createElement("div")
                holder.appendChild(mathjax.tex2chtml(tex))
                before = holder.innerHTML
                self.safety.sanitize(holder)
                # Only the "sanitised" marker is added to the container.
                holder.querySelector("mjx-container").removeAttribute("data-mathud-sanitized")
                self.assertEqual(holder.innerHTML, before)

    def test_skip_sanitized_leaves_marked_containers_alone(self) -> None:
        marked = self._container('<mjx-mi style="position: fixed">a</mjx-mi>')
        self.assertEqual(self.safety.sanitize(self.root), 1)
        self.assertTrue(marked.hasAttribute("data-mathud-sanitized"))
        # A later backstop pass (typesetAndSanitize) skips it and cleans only new output.
        fresh = self._container('<mjx-mi style="position: fixed">b</mjx-mi>')
        self.assertEqual(self.safety.sanitize(self.root, True), 1)
        self.assertFalse(fresh.querySelector("mjx-mi").hasAttribute("style"))
        # A full pass (the render action's) sanitises marked containers again.
        self.assertEqual(self.safety.sanitize(self.root), 2)

    def test_render_math_goes_through_the_sanitizer(self) -> None:
        from chat_ui_manager import ChatUIManager
        from message_menu_manager import MessageMenuManager

        calls: List[Any] = []
        history = document.createElement("div")

        class _Safety:
            def typesetAndSanitize(self, root: Any) -> None:
                calls.append(root)

        class _UI(ChatUIManager):
            def _chat_history_element(self) -> Any:
                return history

        new_message = document.createElement("div")
        original = window.MatHudMathSafety
        window.MatHudMathSafety = _Safety()
        try:
            ui = _UI(message_menu=MessageMenuManager(), tool_call_log=ToolCallLogManager())
            ui.render_math()
            ui.render_math(new_message)  # a message just added: only it is typeset
        finally:
            window.MatHudMathSafety = original
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0], history)
        self.assertIs(calls[1], new_message)


# Math that tex-mml-chtml.js typesets without loading anything.
BUNDLED_MATH = (
    "\\frac{1}{2}",
    "\\sqrt{x^2+1}",
    "\\begin{pmatrix}1 & 2 \\\\ 3 & 4\\end{pmatrix}",
    "\\text{area} = \\pi r^2",
    "\\mathbb{R}",
)
# (extension, TeX) for math whose extension is autoloaded; the page preloads these.
EXTENSION_MATH = (
    ("color", "\\color{red}{x} + \\textcolor{blue}{y}"),
    ("boldsymbol", "\\boldsymbol{v}"),
    ("cancel", "\\cancel{x}"),
    ("bbox", "\\bbox[yellow]{x}"),
    ("bbox", "\\bbox[5px]{x}"),
    ("bbox", "\\bbox[5px,border:2px solid red]{x}"),
    ("unicode", "\\unicode[Arial]{x41}"),
)


def _extension_loaded(mathjax: Any, name: str) -> bool:
    """Whether TeX extension ``name`` has loaded (its component registers under MathJax._.input.tex)."""
    try:
        return bool(getattr(mathjax._.input.tex, name, None))
    except Exception:
        return False


def _ready_mathjax() -> Any:
    """window.MathJax once it can typeset (tex2chtml exists), else None."""
    mathjax = getattr(window, "MathJax", None)
    if mathjax is None or not hasattr(mathjax, "tex2chtml"):
        return None
    return mathjax


def _styles_under(root: Any) -> str:
    """All inline style text of the MathJax output under root."""
    return " ".join(str(element.getAttribute("style") or "") for element in root.querySelectorAll("[style]"))
