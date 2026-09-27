"""Security settings in templates/index.html."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

INDEX_HTML = Path(__file__).resolve().parent.parent / "templates" / "index.html"
STATIC_DIR = INDEX_HTML.parent.parent / "static"


def _mathjax_config() -> str:
    """The inline <script> block that configures MathJax."""
    template = INDEX_HTML.read_text(encoding="utf-8")
    config_at = template.index("window.MathJax = {")
    start = template.rindex("<script>", 0, config_at)
    return template[start : template.index("</script>", config_at)]


class TestMathJaxConfig(unittest.TestCase):
    """MathJax's html extension (\\href, \\class, \\style, \\cssId) must stay off.

    With it, ``\\href{javascript:...}{...}`` in a chat message (live or loaded from a
    workspace) typesets to a clickable ``javascript:`` link.
    """

    def test_html_extension_is_not_autoloaded(self) -> None:
        config = _mathjax_config()
        self.assertRegex(config, r"tex:\s*\{")
        self.assertIsNotNone(re.search(r"autoload:\s*\{\s*html:\s*\[\s*\]\s*[,}]", config), config)

    def test_safe_extension_is_loaded_from_vendor(self) -> None:
        """ui/safe filters href/style/class/id that base TeX (\\mmlToken, \\bbox) can still set."""
        config = _mathjax_config()
        self.assertIsNotNone(re.search(r"loader:\s*\{\s*load:\s*\[\s*'ui/safe'\s*\]\s*\}", config), config)
        # [mathjax] resolves to the folder of tex-mml-chtml.js, so ui/safe must sit next to it.
        vendored = INDEX_HTML.parent.parent / "static" / "vendor" / "mathjax" / "3.2.2" / "es5"
        self.assertTrue((vendored / "tex-mml-chtml.js").is_file())
        self.assertTrue((vendored / "ui" / "safe.js").is_file())
        self.assertNotRegex(config, r"paths\s*:", "a custom path would stop ui/safe resolving to the vendor folder")

    def test_require_and_newcommand_packages_are_removed(self) -> None:
        """No \\require{html}, and no user macros (\\def amplification, lasting redefinitions)."""
        config = _mathjax_config()
        self.assertIsNotNone(
            re.search(r"packages:\s*\{\s*'\[-\]':\s*\[\s*'require',\s*'newcommand'\s*\]\s*\}", config),
            config,
        )

    def test_user_macros_cannot_come_back_through_autoload_or_dependencies(self) -> None:
        config = _mathjax_config()
        self.assertIsNotNone(
            re.search(r"autoload:\s*\{\s*html:\s*\[\s*\],\s*newcommand:\s*\[\s*\],\s*extpfeil:\s*\[\s*\]\s*\}", config),
            config,
        )
        self.assertRegex(config, r"maxMacros:\s*1000")

    def test_macro_and_state_defining_commands_are_removed_from_the_command_maps(self) -> None:
        """\\def & co., \\DeclareMathOperator (AMS, always loaded) and \\definecolor persist or amplify."""
        config = _mathjax_config()
        removed = config[config.index("var MATHUD_REMOVED_COMMANDS = {") :]
        removed = removed[: removed.index("};")]
        self.assertIn(
            "'Newcommand-macros': ['def', 'let', 'newcommand', 'renewcommand', 'newenvironment', 'renewenvironment']",
            removed,
        )
        self.assertIn("'AMSmath-macros': ['DeclareMathOperator']", removed)
        self.assertIn("'color': ['definecolor']", removed)
        self.assertIn("maps.getMap('ams-declare-ops')", config)
        # Removed at startup, after the extension preload, and before every formula.
        self.assertGreaterEqual(config.count("mathudRemoveCommands()"), 2)
        self.assertIn(".then(mathudRemoveCommands)", config)
        # No labels carried from one formula to the next.
        self.assertIn("jax.parseOptions.tags.reset();", config)

    def test_unicode_font_option_is_dropped(self) -> None:
        """MathJax's \\unicode caches each code point's font across formulas (module-private)."""
        config = _mathjax_config()
        self.assertIn("function mathudUnicode(parser, name) {", config)
        self.assertIn(
            "map.map.set('unicode', new MathJax._.input.tex.Symbol.Macro('unicode', mathudUnicode, []));", config
        )
        # Installed wherever the other commands are removed (startup, after the preload,
        # before every formula), so it also applies when \unicode is autoloaded later.
        removal = config[config.index("function mathudRemoveCommands() {") :]
        self.assertIn("mathudReplaceUnicode();", removal[: removal.index("}")])

    def test_long_formulas_are_capped(self) -> None:
        config = _mathjax_config()
        self.assertIn("var MATHUD_MAX_TEX_CHARS = 4000;", config)
        self.assertIn("jax.preFilters.add(", config)

    def test_sanitizer_runs_as_a_render_action(self) -> None:
        """Re-renders (e.g. from the MathJax context menu) must go through the sanitiser too."""
        config = _mathjax_config()
        self.assertIn("renderActions: {", config)
        self.assertIn("mathudSanitize: [", config)
        self.assertIn("var MATHUD_SANITIZE_PRIORITY = 201;", config)
        self.assertIn("window.MatHudMathSafety.sanitize(math.typesetRoot)", config)
        # Already-sanitised formulas are skipped by state; MathJax resets it on re-render.
        self.assertIn("MathJax._.core.MathItem.newState('MATHUDSANITIZED', MATHUD_SANITIZE_PRIORITY);", config)
        self.assertIn("if (item.state() < sanitized) {", config)

    def test_mathml_input_is_removed(self) -> None:
        """It re-typeset the hidden assistive MathML of every formula on each chat typeset."""
        self.assertIn("return jax.name !== 'MathML';", _mathjax_config())

    def test_menu_settings_needing_unvendored_components_are_dropped(self) -> None:
        """A stored {"renderer": "SVG"} (output/svg.js is not vendored) made every typeset hang."""
        config = _mathjax_config()
        kept = config[config.index("var MATHUD_MENU_SETTINGS_KEPT = [") :]
        kept = kept[: kept.index("];")]
        for setting in ("renderer", "explorer", "collapsible", "autocollapse", "locale"):
            self.assertNotIn(f"'{setting}'", kept)
        self.assertIn("var key = 'MathJax-Menu-Settings';", config)
        hidden = config[config.index("var MATHUD_MENU_HIDDEN = [") :]
        hidden = hidden[: hidden.index("];")]
        for entry in ("'Renderer'", "'Language'", "'Activate'", "'Collapsible'", "'AutoCollapse'"):
            self.assertIn(entry, hidden)
        self.assertIn("mathudHideMenuEntries();", config)

    def test_chat_math_is_contained_to_its_own_box(self) -> None:
        css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
        inline = css[css.index('#chat-history mjx-container[jax="CHTML"]:not([display="true"])') :]
        self.assertIn("contain: paint;", inline[: inline.index("}")])
        self.assertIn("overflow-clip-margin: 0.5em;", inline[: inline.index("}")])
        display = css[css.index('#chat-history mjx-container[jax="CHTML"][display="true"]') :]
        block = display[: display.index("}")]
        self.assertIn("contain: paint;", block)
        self.assertIn("overflow-x: auto;", block)

    def test_safe_options_allow_no_urls_and_no_cursor(self) -> None:
        """ui/safe's protocol check misses a tab inside "javascript:"; no URLs at all is safe."""
        config = _mathjax_config()
        self.assertIsNotNone(re.search(r"allow:\s*\{\s*URLs:\s*'none'\s*\}", config), config)
        self.assertIsNotNone(re.search(r"safeStyles:\s*\{\s*cursor:\s*false\s*\}", config), config)

    def test_fontfamily_is_filtered(self) -> None:
        """fontfamily (\\mmlToken, \\unicode's font) is pasted into the style unless filtered."""
        config = _mathjax_config()
        self.assertIn("safe.filterAttributes.set('fontfamily', 'filterFontFamily')", config)
        self.assertIn("var MATHUD_SAFE_FONT_FAMILY = /^[\\w\\s,'\"-]*$/;", config)
        # The filter is installed after the default startup has created the document.
        self.assertLess(config.index("MathJax.startup.defaultReady()"), config.index("filterAttributes.set"))

    def test_page_is_not_typeset_outside_the_sanitised_chat_path(self) -> None:
        self.assertRegex(_mathjax_config(), r"typeset:\s*false")

    def test_output_sanitizer_is_loaded(self) -> None:
        template = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn("url_for('static', filename='math_output_sanitizer.js')", template)
        self.assertTrue((STATIC_DIR / "math_output_sanitizer.js").is_file())


if __name__ == "__main__":
    unittest.main()
