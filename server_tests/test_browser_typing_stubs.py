"""Acceptance tests for Brython browser module type stubs.

Validates that the .pyi stubs under static/client/typing/browser/
are syntactically valid, export the expected names, and pass MyPy
type-checking for common browser API usage patterns.
"""

from __future__ import annotations

import ast
import glob
import os
import tempfile
import unittest

from mypy import api


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STUBS_DIR: str = os.path.join(_ROOT, "static", "client", "typing", "browser")
_MYPY_PATH: str = os.path.join(_ROOT, "static", "client", "typing")

_INIT_PYI: str = os.path.join(_STUBS_DIR, "__init__.pyi")
_DOM_PYI: str = os.path.join(_STUBS_DIR, "_dom.pyi")
_AJAX_PYI: str = os.path.join(_STUBS_DIR, "ajax.pyi")
_AIO_PYI: str = os.path.join(_STUBS_DIR, "aio.pyi")

_ALL_STUB_FILES: list[str] = [_INIT_PYI, _DOM_PYI, _AJAX_PYI, _AIO_PYI]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_mypy_snippet(snippet: str) -> tuple[str, str, int]:
    """Run mypy on a code snippet with browser stubs available.

    Sets ``MYPYPATH`` so mypy discovers the browser stub package.
    """
    fd, path = tempfile.mkstemp(suffix=".py")
    old_mypypath = os.environ.get("MYPYPATH")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(snippet)
        os.environ["MYPYPATH"] = _MYPY_PATH
        result: tuple[str, str, int] = api.run(
            [
                "--python-version",
                "3.11",
                "--no-error-summary",
                "--ignore-missing-imports",
                "--follow-imports",
                "skip",
                "--explicit-package-bases",
                path,
            ]
        )
        return result
    finally:
        if old_mypypath is None:
            os.environ.pop("MYPYPATH", None)
        else:
            os.environ["MYPYPATH"] = old_mypypath
        os.unlink(path)


def _parse_class_names(tree: ast.Module, class_name: str) -> set[str]:
    """Extract method and attribute names from a class in an AST."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    names.add(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    names.add(item.target.id)
    return names


def _get_all_class_names(tree: ast.Module) -> set[str]:
    """Extract all class names defined at module level in an AST."""
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
    return names


def _get_all_function_names(tree: ast.Module) -> set[str]:
    """Extract all function names defined at module level in an AST."""
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
    return names


# ---------------------------------------------------------------------------
# Tests: Stub file validity
# ---------------------------------------------------------------------------


class TestStubFileValidity(unittest.TestCase):
    """Verify that stub files exist, parse correctly, and contain no .py files."""

    def test_stub_files_are_valid_python(self) -> None:
        """Each .pyi file must parse as valid Python."""
        for pyi_path in _ALL_STUB_FILES:
            with self.subTest(file=os.path.basename(pyi_path)):
                with open(pyi_path) as f:
                    source = f.read()
                try:
                    ast.parse(source)
                except SyntaxError as exc:
                    self.fail(f"{pyi_path} has a syntax error: {exc}")

    def test_all_stub_files_exist(self) -> None:
        """All expected stub files must exist on disk."""
        for pyi_path in _ALL_STUB_FILES:
            with self.subTest(file=os.path.basename(pyi_path)):
                self.assertTrue(
                    os.path.isfile(pyi_path),
                    f"Stub file does not exist: {pyi_path}",
                )

    def test_no_runtime_py_files_in_stubs(self) -> None:
        """The stubs directory must contain only .pyi files, no .py files."""
        py_files = glob.glob(os.path.join(_STUBS_DIR, "*.py"))
        self.assertEqual(
            py_files,
            [],
            f"Found .py files in stubs directory (should be .pyi only): {py_files}",
        )

    def test_stubs_directory_is_a_package(self) -> None:
        """The browser stubs directory must have an __init__.pyi file."""
        self.assertTrue(
            os.path.isfile(_INIT_PYI),
            "__init__.pyi must exist for the browser stub package",
        )

    def test_all_stubs_have_future_annotations(self) -> None:
        """Each stub file must use ``from __future__ import annotations``."""
        for pyi_path in _ALL_STUB_FILES:
            with self.subTest(file=os.path.basename(pyi_path)):
                with open(pyi_path) as f:
                    tree = ast.parse(f.read())
                has_future = False
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                        for alias in node.names:
                            if alias.name == "annotations":
                                has_future = True
                self.assertTrue(
                    has_future,
                    f"{os.path.basename(pyi_path)} missing 'from __future__ import annotations'",
                )

    def test_all_stubs_have_docstrings(self) -> None:
        """Each stub file must have a module-level docstring."""
        for pyi_path in _ALL_STUB_FILES:
            with self.subTest(file=os.path.basename(pyi_path)):
                with open(pyi_path) as f:
                    tree = ast.parse(f.read())
                docstring = ast.get_docstring(tree)
                self.assertIsNotNone(
                    docstring,
                    f"{os.path.basename(pyi_path)} missing module docstring",
                )

    def test_old_single_file_stub_removed(self) -> None:
        """The obsolete single-file browser.pyi must no longer exist."""
        old_stub = os.path.join(_ROOT, "static", "client", "browser.pyi")
        self.assertFalse(
            os.path.exists(old_stub),
            f"Obsolete stub still exists and should be deleted: {old_stub}",
        )


# ---------------------------------------------------------------------------
# Tests: Export completeness
# ---------------------------------------------------------------------------


class TestExportCompleteness(unittest.TestCase):
    """Verify that stub packages export all names used by the codebase."""

    def test_all_expected_names_exported(self) -> None:
        """__init__.pyi must export all 7 module-level names plus DOMNode and ClassList."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        names: set[str] = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    exported = alias.asname if alias.asname else alias.name
                    names.add(exported)
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)

        expected = {"document", "window", "html", "svg", "console", "ajax", "aio", "DOMNode", "ClassList"}
        missing = expected - names
        self.assertFalse(missing, f"Missing exports in __init__.pyi: {missing}")

    def test_reexports_use_explicit_as_syntax(self) -> None:
        """Re-exports must use ``X as X`` syntax for implicit_reexport=False compat."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        reexport_names = {"DOMNode", "ClassList", "ajax", "aio"}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in reexport_names:
                        with self.subTest(name=alias.name):
                            self.assertEqual(
                                alias.asname,
                                alias.name,
                                f"Re-export '{alias.name}' must use 'as {alias.name}' "
                                f"for implicit_reexport=False compatibility",
                            )

    def test_init_defines_public_classes(self) -> None:
        """__init__.pyi must define Document, Window, HTMLFactory, SVGFactory, Console."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        classes = _get_all_class_names(tree)
        expected = {"Document", "Window", "HTMLFactory", "SVGFactory", "Console"}
        missing = expected - classes
        self.assertFalse(missing, f"Missing public classes: {missing}")

    def test_init_defines_helper_classes(self) -> None:
        """__init__.pyi must define private helper classes for typed Window sub-objects."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        classes = _get_all_class_names(tree)
        expected_helpers = {
            "_JSON",
            "_LocalStorage",
            "_Performance",
            "_MathJS",
            "_NerdamerExpr",
            "_Nerdamer",
            "_Date",
            "_URL",
        }
        missing = expected_helpers - classes
        self.assertFalse(missing, f"Missing helper classes: {missing}")

    def test_module_level_instances_annotated(self) -> None:
        """Module-level singletons must be annotated with their class types."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        annotations: dict[str, str] = {}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if isinstance(node.annotation, ast.Name):
                    annotations[node.target.id] = node.annotation.id

        expected = {
            "document": "Document",
            "window": "Window",
            "html": "HTMLFactory",
            "svg": "SVGFactory",
            "console": "Console",
        }
        for name, expected_type in expected.items():
            with self.subTest(name=name):
                self.assertIn(name, annotations, f"Missing annotation for '{name}'")
                self.assertEqual(
                    annotations[name],
                    expected_type,
                    f"'{name}' should be annotated as {expected_type}, got {annotations.get(name)}",
                )


# ---------------------------------------------------------------------------
# Tests: DOMNode structure
# ---------------------------------------------------------------------------


class TestDOMNodeStructure(unittest.TestCase):
    """Verify DOMNode and ClassList class structures in _dom.pyi."""

    def test_dom_node_has_required_methods(self) -> None:
        """DOMNode class must have all expected methods."""
        with open(_DOM_PYI) as f:
            tree = ast.parse(f.read())

        methods = _parse_class_names(tree, "DOMNode")
        expected_methods = {
            "getBoundingClientRect",
            "appendChild",
            "removeChild",
            "setAttribute",
            "getAttribute",
            "removeAttribute",
            "insertBefore",
            "cloneNode",
            "bind",
            "focus",
            "blur",
            "click",
            "clear",
            "remove",
            "getContext",
            "select",
            "select_one",
            "__getitem__",
            "__setitem__",
            "__le__",
            "__contains__",
        }
        missing = expected_methods - methods
        self.assertFalse(missing, f"DOMNode missing methods: {missing}")

    def test_dom_node_has_required_properties(self) -> None:
        """DOMNode class must have all expected properties."""
        with open(_DOM_PYI) as f:
            tree = ast.parse(f.read())

        attrs = _parse_class_names(tree, "DOMNode")
        expected_props = {
            "innerHTML",
            "text",
            "value",
            "disabled",
            "checked",
            "options",
            "scrollTop",
            "scrollHeight",
            "style",
            "classList",
            "attrs",
            "parentNode",
            "children",
            "firstChild",
            "width",
            "height",
            "onload",
            "result",
            "responseType",
        }
        missing = expected_props - attrs
        self.assertFalse(missing, f"DOMNode missing properties: {missing}")

    def test_classlist_has_required_methods(self) -> None:
        """ClassList class must have add, remove, contains methods."""
        with open(_DOM_PYI) as f:
            tree = ast.parse(f.read())

        methods = _parse_class_names(tree, "ClassList")
        expected = {"add", "remove", "contains"}
        missing = expected - methods
        self.assertFalse(missing, f"ClassList missing methods: {missing}")

    def test_dom_node_classlist_typed_as_classlist(self) -> None:
        """DOMNode.classList must be annotated as ClassList type."""
        with open(_DOM_PYI) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DOMNode":
                for item in node.body:
                    if (
                        isinstance(item, ast.AnnAssign)
                        and isinstance(item.target, ast.Name)
                        and item.target.id == "classList"
                    ):
                        self.assertIsInstance(item.annotation, ast.Name)
                        assert isinstance(item.annotation, ast.Name)
                        self.assertEqual(item.annotation.id, "ClassList")
                        return
        self.fail("DOMNode.classList annotation not found")

    def test_dom_node_parent_typed_as_optional(self) -> None:
        """DOMNode.parentNode must be typed as DOMNode | None."""
        with open(_DOM_PYI) as f:
            source = f.read()
        self.assertIn("parentNode: DOMNode | None", source)

    def test_dom_node_firstchild_typed_as_optional(self) -> None:
        """DOMNode.firstChild must be typed as DOMNode | None."""
        with open(_DOM_PYI) as f:
            source = f.read()
        self.assertIn("firstChild: DOMNode | None", source)


# ---------------------------------------------------------------------------
# Tests: Ajax module structure
# ---------------------------------------------------------------------------


class TestAjaxModuleStructure(unittest.TestCase):
    """Verify ajax.pyi class and function structures."""

    def test_ajax_module_has_required_symbols(self) -> None:
        """ajax.pyi must define AjaxRequest, Ajax, ajax, post."""
        with open(_AJAX_PYI) as f:
            tree = ast.parse(f.read())

        names: set[str] = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.FunctionDef):
                names.add(node.name)

        expected = {"AjaxRequest", "Ajax", "ajax", "post"}
        missing = expected - names
        self.assertFalse(missing, f"ajax.pyi missing symbols: {missing}")

    def test_ajax_request_has_required_members(self) -> None:
        """AjaxRequest class must have status, text, response, responseType, bind, open, set_header, send."""
        with open(_AJAX_PYI) as f:
            tree = ast.parse(f.read())

        members = _parse_class_names(tree, "AjaxRequest")
        expected = {"status", "text", "response", "responseType", "bind", "open", "set_header", "send"}
        missing = expected - members
        self.assertFalse(missing, f"AjaxRequest missing members: {missing}")

    def test_ajax_class_has_same_interface_as_request(self) -> None:
        """Ajax class must have the same interface as AjaxRequest."""
        with open(_AJAX_PYI) as f:
            tree = ast.parse(f.read())

        request_members = _parse_class_names(tree, "AjaxRequest")
        ajax_members = _parse_class_names(tree, "Ajax")
        missing = request_members - ajax_members
        self.assertFalse(
            missing,
            f"Ajax class missing members present in AjaxRequest: {missing}",
        )

    def test_ajax_is_separate_class_not_alias(self) -> None:
        """Ajax must be a separate ClassDef, not a type alias."""
        with open(_AJAX_PYI) as f:
            tree = ast.parse(f.read())

        ajax_classes = [
            node for node in ast.iter_child_nodes(tree) if isinstance(node, ast.ClassDef) and node.name == "Ajax"
        ]
        self.assertEqual(
            len(ajax_classes),
            1,
            "Ajax must be defined as a separate class (not a type alias)",
        )


# ---------------------------------------------------------------------------
# Tests: Aio module structure
# ---------------------------------------------------------------------------


class TestAioModuleStructure(unittest.TestCase):
    """Verify aio.pyi function structures."""

    def test_aio_module_has_required_symbols(self) -> None:
        """aio.pyi must define run and sleep."""
        with open(_AIO_PYI) as f:
            tree = ast.parse(f.read())

        names = _get_all_function_names(tree)
        expected = {"run", "sleep"}
        missing = expected - names
        self.assertFalse(missing, f"aio.pyi missing symbols: {missing}")

    def test_aio_has_no_classes(self) -> None:
        """aio.pyi should only have functions, no classes."""
        with open(_AIO_PYI) as f:
            tree = ast.parse(f.read())

        classes = _get_all_class_names(tree)
        self.assertEqual(classes, set(), f"aio.pyi should not define classes: {classes}")


# ---------------------------------------------------------------------------
# Tests: Window class structure
# ---------------------------------------------------------------------------


class TestWindowStructure(unittest.TestCase):
    """Verify Window class structure in __init__.pyi."""

    def test_window_has_typed_sub_objects(self) -> None:
        """Window must have typed sub-object annotations."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        members = _parse_class_names(tree, "Window")
        expected = {
            "JSON",
            "localStorage",
            "performance",
            "math",
            "nerdamer",
            "Date",
            "URL",
        }
        missing = expected - members
        self.assertFalse(missing, f"Window missing typed sub-objects: {missing}")

    def test_window_has_direct_methods(self) -> None:
        """Window must have setTimeout, clearTimeout, requestAnimationFrame."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        members = _parse_class_names(tree, "Window")
        expected = {"setTimeout", "clearTimeout", "requestAnimationFrame"}
        missing = expected - members
        self.assertFalse(missing, f"Window missing direct methods: {missing}")

    def test_window_has_constructor_attributes(self) -> None:
        """Window must have constructor-like attributes (Audio, Float32Array, etc.)."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        members = _parse_class_names(tree, "Window")
        expected = {"Audio", "Float32Array", "FileReader", "MouseEvent", "MathJax", "Math"}
        missing = expected - members
        self.assertFalse(missing, f"Window missing constructor attributes: {missing}")

    def test_window_has_escape_hatches(self) -> None:
        """Window must define __getattr__, __setattr__, __getitem__, __setitem__."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        members = _parse_class_names(tree, "Window")
        expected = {"__getattr__", "__setattr__", "__getitem__", "__setitem__"}
        missing = expected - members
        self.assertFalse(missing, f"Window missing escape hatches: {missing}")


# ---------------------------------------------------------------------------
# Tests: Document class structure
# ---------------------------------------------------------------------------


class TestDocumentStructure(unittest.TestCase):
    """Verify Document class structure in __init__.pyi."""

    def test_document_has_query_methods(self) -> None:
        """Document must have getElementById, querySelector, select, select_one."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        members = _parse_class_names(tree, "Document")
        expected = {"getElementById", "querySelector", "select", "select_one"}
        missing = expected - members
        self.assertFalse(missing, f"Document missing query methods: {missing}")

    def test_document_has_operators(self) -> None:
        """Document must define __getitem__, __contains__, __le__."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        members = _parse_class_names(tree, "Document")
        expected = {"__getitem__", "__contains__", "__le__"}
        missing = expected - members
        self.assertFalse(missing, f"Document missing operators: {missing}")

    def test_document_has_bind_method(self) -> None:
        """Document must define bind for event handling."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        members = _parse_class_names(tree, "Document")
        self.assertIn("bind", members)

    def test_document_not_subclass_of_domnode(self) -> None:
        """Document must NOT be a subclass of DOMNode to avoid Liskov conflicts."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Document":
                domnode_bases = [b for b in node.bases if isinstance(b, ast.Name) and b.id == "DOMNode"]
                self.assertEqual(
                    len(domnode_bases),
                    0,
                    "Document must not inherit from DOMNode (Liskov conflict)",
                )
                return
        self.fail("Document class not found")


# ---------------------------------------------------------------------------
# Tests: HTMLFactory and SVGFactory structure
# ---------------------------------------------------------------------------


class TestFactoryStructure(unittest.TestCase):
    """Verify HTMLFactory and SVGFactory class structures."""

    def test_html_factory_has_element_methods(self) -> None:
        """HTMLFactory must define methods for all used HTML elements."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        members = _parse_class_names(tree, "HTMLFactory")
        expected = {
            "DIV",
            "SPAN",
            "BUTTON",
            "INPUT",
            "TEXTAREA",
            "LABEL",
            "SELECT",
            "OPTION",
            "H3",
            "P",
            "IMG",
            "CANVAS",
            "DETAILS",
            "SUMMARY",
        }
        missing = expected - members
        self.assertFalse(missing, f"HTMLFactory missing element methods: {missing}")

    def test_html_factory_has_getattr_fallback(self) -> None:
        """HTMLFactory must define __getattr__ for unlisted elements."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        members = _parse_class_names(tree, "HTMLFactory")
        self.assertIn("__getattr__", members)

    def test_svg_factory_has_element_methods(self) -> None:
        """SVGFactory must define methods for all used SVG elements."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        members = _parse_class_names(tree, "SVGFactory")
        expected = {"svg", "g", "line", "path", "circle", "ellipse", "polygon", "text"}
        missing = expected - members
        self.assertFalse(missing, f"SVGFactory missing element methods: {missing}")

    def test_svg_factory_has_getattr_fallback(self) -> None:
        """SVGFactory must define __getattr__ for unlisted elements."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        members = _parse_class_names(tree, "SVGFactory")
        self.assertIn("__getattr__", members)


# ---------------------------------------------------------------------------
# Tests: Console structure
# ---------------------------------------------------------------------------


class TestConsoleStructure(unittest.TestCase):
    """Verify Console class structure."""

    def test_console_has_logging_methods(self) -> None:
        """Console must have log, error, warn, groupCollapsed, groupEnd."""
        with open(_INIT_PYI) as f:
            tree = ast.parse(f.read())

        members = _parse_class_names(tree, "Console")
        expected = {"log", "error", "warn", "groupCollapsed", "groupEnd"}
        missing = expected - members
        self.assertFalse(missing, f"Console missing methods: {missing}")


# ---------------------------------------------------------------------------
# Tests: MyPy resolution and type-checking
# ---------------------------------------------------------------------------


class TestMyPyResolution(unittest.TestCase):
    """Verify MyPy can resolve browser imports and type-check operations."""

    def test_mypy_resolves_browser_imports(self) -> None:
        """MyPy must resolve all browser imports without errors."""
        snippet = "from browser import document, window, html, svg, ajax, aio, console\n"
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")
        self.assertNotIn(
            "Cannot find implementation or library stub",
            stdout,
            f"MyPy could not find browser stubs:\n{stdout}",
        )

    def test_mypy_resolves_domnode_import(self) -> None:
        """MyPy must resolve DOMNode and ClassList from browser._dom."""
        snippet = "from browser import DOMNode, ClassList\nreveal_type(DOMNode)\nreveal_type(ClassList)\n"
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_reveal_types(self) -> None:
        """reveal_type() must produce expected stub types."""
        snippet = (
            "from browser import document, window, html, svg, console\n"
            "reveal_type(document)\n"
            "reveal_type(window)\n"
            "reveal_type(html)\n"
            "reveal_type(svg)\n"
            "reveal_type(console)\n"
        )
        stdout, _stderr, _exit_code = _run_mypy_snippet(snippet)
        self.assertIn("Document", stdout)
        self.assertIn("Window", stdout)
        self.assertIn("HTMLFactory", stdout)
        self.assertIn("SVGFactory", stdout)
        self.assertIn("Console", stdout)

    def test_mypy_reveal_ajax_types(self) -> None:
        """reveal_type() for ajax objects must show AjaxRequest/Ajax."""
        snippet = "from browser import ajax\nreq = ajax.ajax()\nreveal_type(req)\n"
        stdout, _stderr, _exit_code = _run_mypy_snippet(snippet)
        self.assertIn("AjaxRequest", stdout)

    def test_stubs_pass_mypy(self) -> None:
        """MyPy must accept the stub files themselves without errors."""
        old_mypypath = os.environ.get("MYPYPATH")
        try:
            os.environ["MYPYPATH"] = _MYPY_PATH
            result = api.run(
                [
                    "--python-version",
                    "3.11",
                    "--no-error-summary",
                    "--ignore-missing-imports",
                    "--explicit-package-bases",
                    *_ALL_STUB_FILES,
                ]
            )
        finally:
            if old_mypypath is None:
                os.environ.pop("MYPYPATH", None)
            else:
                os.environ["MYPYPATH"] = old_mypypath
        stdout, stderr, exit_code = result
        self.assertEqual(exit_code, 0, f"MyPy failed on stubs:\nstdout: {stdout}\nstderr: {stderr}")


# ---------------------------------------------------------------------------
# Tests: MyPy DOM operations
# ---------------------------------------------------------------------------


class TestMyPyDOMOperations(unittest.TestCase):
    """Verify MyPy type-checks common DOM operation patterns."""

    def test_mypy_basic_dom_operations(self) -> None:
        """Basic DOM operations must type-check without errors."""
        snippet = (
            "from browser import document, window, html\n"
            "el = document.getElementById('x')\n"
            "timer = window.setTimeout(lambda: None, 100)\n"
            "div = html.DIV(Class='foo')\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_document_subscript_access(self) -> None:
        """document['id'] must type-check for element access by ID."""
        snippet = "from browser import document\nel = document['chat-input']\nel.value = 'hello'\n"
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_document_contains(self) -> None:
        """'id' in document must type-check for element existence checks."""
        snippet = "from browser import document\nif 'run-tests-button' in document:\n    pass\n"
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_dom_append_operator(self) -> None:
        """document <= element (Brython DOM append) must type-check."""
        snippet = "from browser import document, html\ndiv = html.DIV()\ndocument <= div\n"
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_element_append_operator(self) -> None:
        """parent <= child (Brython element append) must type-check."""
        snippet = "from browser import html\nparent = html.DIV()\nchild = html.SPAN('text')\nparent <= child\n"
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_dom_node_methods(self) -> None:
        """DOMNode methods (appendChild, setAttribute, etc.) must type-check."""
        snippet = (
            "from browser import document, html\n"
            "parent = html.DIV()\n"
            "child = html.SPAN()\n"
            "parent.appendChild(child)\n"
            "parent.setAttribute('data-id', '42')\n"
            "val = parent.getAttribute('data-id')\n"
            "parent.removeAttribute('data-id')\n"
            "clone = parent.cloneNode(True)\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_dom_node_properties(self) -> None:
        """DOMNode properties (innerHTML, scrollTop, etc.) must type-check."""
        snippet = (
            "from browser import html\n"
            "el = html.DIV()\n"
            "el.innerHTML = '<b>bold</b>'\n"
            "s: str = el.innerHTML\n"
            "el.style.display = 'none'\n"
            "top: float = el.scrollTop\n"
            "height: float = el.scrollHeight\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_classlist_operations(self) -> None:
        """ClassList operations (add, remove, contains) must type-check."""
        snippet = (
            "from browser import html\n"
            "el = html.DIV()\n"
            "el.classList.add('active')\n"
            "el.classList.remove('hidden')\n"
            "has: bool = el.classList.contains('active')\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_event_binding(self) -> None:
        """Element event binding must type-check."""
        snippet = (
            "from browser import document, html\n"
            "el = html.BUTTON('Click me')\n"
            "el.bind('click', lambda ev: None)\n"
            "document.bind('keydown', lambda ev: None)\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_dom_traversal(self) -> None:
        """DOM node traversal (parentNode, firstChild) must type-check."""
        snippet = (
            "from browser import html, DOMNode\n"
            "el = html.DIV()\n"
            "parent = el.parentNode\n"
            "if parent is not None:\n"
            "    parent.removeChild(el)\n"
            "first = el.firstChild\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_select_methods(self) -> None:
        """DOMNode.select() and select_one() must type-check."""
        snippet = (
            "from browser import html, DOMNode\n"
            "el = html.DIV()\n"
            "results: list[DOMNode] = el.select('.child')\n"
            "one = el.select_one('.child')\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")


# ---------------------------------------------------------------------------
# Tests: MyPy HTML/SVG factory operations
# ---------------------------------------------------------------------------


class TestMyPyFactoryOperations(unittest.TestCase):
    """Verify MyPy type-checks HTML and SVG factory patterns."""

    def test_mypy_html_factory_with_content(self) -> None:
        """HTMLFactory methods with positional content must type-check."""
        snippet = (
            "from browser import html\n"
            "btn = html.BUTTON('Click me', Class='btn-primary')\n"
            "span = html.SPAN('text', Class='highlight')\n"
            "p = html.P('paragraph')\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_html_factory_keyword_only(self) -> None:
        """HTMLFactory methods with keyword-only args must type-check."""
        snippet = (
            "from browser import html\n"
            "inp = html.INPUT(id='user-input')\n"
            "sel = html.SELECT(id='model-selector')\n"
            "canvas = html.CANVAS(id='main-canvas')\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_html_factory_returns_domnode(self) -> None:
        """HTMLFactory methods must return DOMNode-compatible objects."""
        snippet = (
            "from browser import html, DOMNode\n"
            "div = html.DIV()\n"
            "div.appendChild(html.SPAN())\n"
            "div.setAttribute('data-custom', 'val')\n"
            "div.bind('click', lambda ev: None)\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_html_factory_getattr_fallback(self) -> None:
        """HTMLFactory.__getattr__ must allow arbitrary element creation."""
        snippet = "from browser import html\nel = html.ARTICLE(Class='content')\n"
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_svg_factory_operations(self) -> None:
        """SVG factory element creation must type-check."""
        snippet = (
            "from browser import svg\n"
            "root = svg.svg(width='100', height='100')\n"
            "group = svg.g(id='layer1')\n"
            "ln = svg.line(x1='0', y1='0', x2='100', y2='100')\n"
            "p = svg.path(d='M0 0 L10 10')\n"
            "c = svg.circle(cx='50', cy='50', r='25')\n"
            "e = svg.ellipse(cx='50', cy='50', rx='30', ry='20')\n"
            "pg = svg.polygon(points='0,0 10,0 5,10')\n"
            "t = svg.text(x='10', y='20')\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_svg_factory_getattr_fallback(self) -> None:
        """SVGFactory.__getattr__ must allow arbitrary SVG element creation."""
        snippet = "from browser import svg\nrect = svg.rect(x='0', y='0', width='100', height='50')\n"
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")


# ---------------------------------------------------------------------------
# Tests: MyPy Window operations
# ---------------------------------------------------------------------------


class TestMyPyWindowOperations(unittest.TestCase):
    """Verify MyPy type-checks Window object usage patterns."""

    def test_mypy_window_json(self) -> None:
        """window.JSON.stringify/parse must type-check."""
        snippet = (
            "from browser import window\n"
            "from typing import Any\n"
            "s: str = window.JSON.stringify({'key': 'value'})\n"
            "obj: Any = window.JSON.parse(s)\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_window_localstorage(self) -> None:
        """window.localStorage methods must type-check."""
        snippet = (
            "from browser import window\n"
            "val = window.localStorage.getItem('key')\n"
            "window.localStorage.setItem('key', 'value')\n"
            "window.localStorage.removeItem('key')\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_window_performance(self) -> None:
        """window.performance.now() must type-check and return float."""
        snippet = "from browser import window\nt: float = window.performance.now()\n"
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_window_timeout(self) -> None:
        """setTimeout/clearTimeout must type-check with correct types."""
        snippet = (
            "from browser import window\n"
            "timer_id: int = window.setTimeout(lambda: None, 100)\n"
            "window.clearTimeout(timer_id)\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_window_request_animation_frame(self) -> None:
        """requestAnimationFrame must type-check."""
        snippet = "from browser import window\nframe_id: int = window.requestAnimationFrame(lambda ts: None)\n"
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_window_mathjs(self) -> None:
        """window.math (math.js) operations must type-check."""
        snippet = (
            "from browser import window\n"
            "from typing import Any\n"
            "result: Any = window.math.evaluate('2 + 2')\n"
            "formatted: str = window.math.format(result)\n"
            "sq: Any = window.math.sqrt(4)\n"
            "pw: Any = window.math.pow(2, 3)\n"
            "d: Any = window.math.det([[1, 2], [3, 4]])\n"
            "t: str = window.math.typeOf(result)\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_window_nerdamer(self) -> None:
        """window.nerdamer operations must type-check."""
        snippet = (
            "from browser import window\n"
            "expr = window.nerdamer('x^2 + 1')\n"
            "text: str = expr.text()\n"
            "evald = expr.evaluate()\n"
            "subbed = expr.sub('x', 2)\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_window_dynamic_attribute_access(self) -> None:
        """Window __getattr__/__setattr__ escape hatch must type-check."""
        snippet = (
            "from browser import window\n"
            "from typing import Any\n"
            "window.startMatHudTests = lambda: None\n"
            "custom: Any = window.VISION_MODELS\n"
            "window['custom_key'] = 42\n"
            "val = window['custom_key']\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_window_date(self) -> None:
        """window.Date.now() must type-check."""
        snippet = "from browser import window\nts: int = window.Date.now()\n"
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_window_url(self) -> None:
        """window.URL.createObjectURL/revokeObjectURL must type-check."""
        snippet = (
            "from browser import window\n"
            "from typing import Any\n"
            "blob: Any = None\n"
            "url: str = window.URL.createObjectURL(blob)\n"
            "window.URL.revokeObjectURL(url)\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")


# ---------------------------------------------------------------------------
# Tests: MyPy AJAX operations
# ---------------------------------------------------------------------------


class TestMyPyAjaxOperations(unittest.TestCase):
    """Verify MyPy type-checks AJAX usage patterns."""

    def test_mypy_ajax_lowercase_constructor(self) -> None:
        """ajax.ajax() must type-check and return an AjaxRequest."""
        snippet = (
            "from browser import ajax\n"
            "req = ajax.ajax()\n"
            "req.bind('complete', lambda r: None)\n"
            "req.open('GET', '/api')\n"
            "req.send()\n"
            "status: int = req.status\n"
            "body: str = req.text\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_ajax_uppercase_constructor(self) -> None:
        """ajax.Ajax() must type-check as a separate constructor."""
        snippet = (
            "from browser import ajax\n"
            "req = ajax.Ajax()\n"
            "req.open('POST', '/api')\n"
            "req.set_header('Content-Type', 'application/json')\n"
            'req.send(\'{"key": "value"}\')\n'
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_ajax_post_shortcut(self) -> None:
        """ajax.post() shortcut must type-check."""
        snippet = "from browser import ajax\najax.post('/api', data='payload', oncomplete=lambda r: None)\n"
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_ajax_with_timeout(self) -> None:
        """ajax.ajax(timeout=...) must type-check."""
        snippet = "from browser import ajax\nreq = ajax.ajax(timeout=20000)\nreq.open('GET', '/api')\nreq.send()\n"
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_ajax_both_constructors_type_check(self) -> None:
        """Both ajax.ajax() and ajax.Ajax() must produce usable objects."""
        snippet = (
            "from browser import ajax\n"
            "req1 = ajax.ajax()\n"
            "req2 = ajax.Ajax()\n"
            "req1.bind('complete', lambda r: None)\n"
            "req2.bind('complete', lambda r: None)\n"
            "req1.open('GET', '/api')\n"
            "req2.open('POST', '/api')\n"
            "req1.send()\n"
            "req2.send('data')\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")


# ---------------------------------------------------------------------------
# Tests: MyPy async I/O operations
# ---------------------------------------------------------------------------


class TestMyPyAioOperations(unittest.TestCase):
    """Verify MyPy type-checks async I/O patterns."""

    def test_mypy_aio_run_and_sleep(self) -> None:
        """aio.run() and aio.sleep() must type-check."""
        snippet = "from browser import aio\nasync def main() -> None:\n    await aio.sleep(1.0)\naio.run(main())\n"
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_aio_run_accepts_coroutine(self) -> None:
        """aio.run() must accept a coroutine argument."""
        snippet = (
            "from browser import aio\n"
            "async def fetch_data() -> str:\n"
            "    await aio.sleep(0.5)\n"
            "    return 'data'\n"
            "aio.run(fetch_data())\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")


# ---------------------------------------------------------------------------
# Tests: MyPy console operations
# ---------------------------------------------------------------------------


class TestMyPyConsoleOperations(unittest.TestCase):
    """Verify MyPy type-checks console usage patterns."""

    def test_mypy_console_logging(self) -> None:
        """console.log/error/warn with various args must type-check."""
        snippet = (
            "from browser import console\n"
            "console.log('message')\n"
            "console.log('key', 42, [1, 2, 3])\n"
            "console.error('error message')\n"
            "console.warn('warning')\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_console_grouping(self) -> None:
        """console.groupCollapsed/groupEnd must type-check."""
        snippet = (
            "from browser import console\n"
            "console.groupCollapsed('Debug info')\n"
            "console.log('details')\n"
            "console.groupEnd()\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")


# ---------------------------------------------------------------------------
# Tests: Integration patterns (real codebase usage)
# ---------------------------------------------------------------------------


class TestMyPyIntegrationPatterns(unittest.TestCase):
    """Verify MyPy handles real codebase usage patterns from static/client/."""

    def test_mypy_ajax_bind_open_send_pattern(self) -> None:
        """Full AJAX request lifecycle pattern from workspace_manager.py."""
        snippet = (
            "from browser import ajax\n"
            "def save_workspace(name: str, data: str) -> None:\n"
            "    req = ajax.Ajax()\n"
            "    req.bind('complete', lambda r: None)\n"
            "    req.open('POST', '/save_workspace')\n"
            "    req.set_header('Content-Type', 'application/json')\n"
            "    req.send(data)\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_dom_element_creation_and_append(self) -> None:
        """Element creation + append pattern from ai_interface.py."""
        snippet = (
            "from browser import document, html\n"
            "def create_ui() -> None:\n"
            "    container = html.DIV(Class='container')\n"
            "    button = html.BUTTON('Send', Class='send-btn', id='send-button')\n"
            "    textarea = html.TEXTAREA(id='chat-input')\n"
            "    container <= button\n"
            "    container <= textarea\n"
            "    document <= container\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_local_storage_pattern(self) -> None:
        """LocalStorage get/set pattern from tts_controller.py."""
        snippet = (
            "from browser import window\n"
            "STORAGE_KEY = 'mathud.voice'\n"
            "val = window.localStorage.getItem(STORAGE_KEY)\n"
            "if val is not None:\n"
            "    current: str = val\n"
            "window.localStorage.setItem(STORAGE_KEY, 'default')\n"
            "window.localStorage.removeItem(STORAGE_KEY)\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_svg_rendering_pattern(self) -> None:
        """SVG rendering pattern from svg_primitive_adapter.py."""
        snippet = (
            "from browser import svg, document\n"
            "def create_svg_scene() -> None:\n"
            "    root = svg.svg(id='canvas-svg', width='800', height='600')\n"
            "    group = svg.g(id='drawables')\n"
            "    ln = svg.line(x1='0', y1='0', x2='100', y2='100')\n"
            "    c = svg.circle(cx='50', cy='50', r='10')\n"
            "    group <= ln\n"
            "    group <= c\n"
            "    root <= group\n"
            "    frag = document.createDocumentFragment()\n"
            "    frag <= root\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_element_query_and_manipulation(self) -> None:
        """Query + manipulate pattern from canvas_event_handler.py."""
        snippet = (
            "from browser import document\n"
            "def setup_handlers() -> None:\n"
            "    if 'run-tests-button' in document:\n"
            "        btn = document['run-tests-button']\n"
            "        btn.bind('click', lambda ev: None)\n"
            "    el = document.getElementById('main-container')\n"
            "    if el is not None:\n"
            "        el.classList.add('active')\n"
            "        el.classList.remove('hidden')\n"
            "        has: bool = el.classList.contains('active')\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_math_evaluation_pattern(self) -> None:
        """Math evaluation pattern from expression_evaluator.py."""
        snippet = (
            "from browser import window\n"
            "from typing import Any\n"
            "def evaluate_expr(expr: str) -> str:\n"
            "    result: Any = window.math.evaluate(expr)\n"
            "    formatted: str = window.math.format(result)\n"
            "    type_name: str = window.math.typeOf(result)\n"
            "    return formatted\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_nerdamer_solve_pattern(self) -> None:
        """Nerdamer symbolic math pattern from math_utils.py."""
        snippet = (
            "from browser import window\n"
            "def solve_equation(eq: str) -> str:\n"
            "    expr = window.nerdamer(eq)\n"
            "    result = expr.evaluate()\n"
            "    text: str = result.text()\n"
            "    subbed = expr.sub('x', 5)\n"
            "    return text\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_async_data_fetch_pattern(self) -> None:
        """Async fetch pattern from ai_interface.py."""
        snippet = (
            "from browser import aio, ajax\n"
            "async def fetch_response() -> None:\n"
            "    await aio.sleep(0.1)\n"
            "    req = ajax.ajax()\n"
            "    req.bind('complete', lambda r: None)\n"
            "    req.open('GET', '/api/status')\n"
            "    req.send()\n"
            "aio.run(fetch_response())\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_canvas_getcontext_pattern(self) -> None:
        """Canvas getContext pattern from canvas2d_renderer.py."""
        snippet = (
            "from browser import html\n"
            "from typing import Any\n"
            "canvas = html.CANVAS(id='main-canvas')\n"
            "canvas.attrs['width'] = '800'\n"
            "canvas.attrs['height'] = '600'\n"
            "ctx: Any = canvas.getContext('2d')\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_json_stringify_parse_round_trip(self) -> None:
        """JSON round-trip pattern from linear_algebra_utils.py."""
        snippet = (
            "from browser import window\n"
            "from typing import Any\n"
            "data: dict[str, int] = {'x': 1, 'y': 2}\n"
            "json_str: str = window.JSON.stringify(data)\n"
            "parsed: Any = window.JSON.parse(json_str)\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_performance_timing_pattern(self) -> None:
        """Performance timing pattern from rendering code."""
        snippet = (
            "from browser import window\n"
            "start: float = window.performance.now()\n"
            "# ... do work ...\n"
            "end: float = window.performance.now()\n"
            "elapsed: float = end - start\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_window_function_registration_pattern(self) -> None:
        """Dynamic window function registration pattern from main.py."""
        snippet = (
            "from browser import window\n"
            "from typing import Any\n"
            "def start_tests() -> dict[str, str]:\n"
            "    return {'status': 'started'}\n"
            "def get_results() -> dict[str, Any]:\n"
            "    return {'tests_run': 0}\n"
            "window.startMatHudTests = start_tests\n"
            "window.getMatHudTestResults = get_results\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_document_exec_command(self) -> None:
        """document.execCommand pattern from ai_interface.py."""
        snippet = "from browser import document\nresult: bool = document.execCommand('copy')\n"
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")

    def test_mypy_insert_before_pattern(self) -> None:
        """insertBefore pattern from svg_primitive_adapter.py."""
        snippet = (
            "from browser import svg\n"
            "surface = svg.svg()\n"
            "group = svg.g()\n"
            "ref = svg.g()\n"
            "surface.insertBefore(group, ref)\n"
            "surface.insertBefore(group, None)\n"
        )
        stdout, stderr, exit_code = _run_mypy_snippet(snippet)
        self.assertEqual(exit_code, 0, f"MyPy failed:\nstdout: {stdout}\nstderr: {stderr}")


# ---------------------------------------------------------------------------
# Tests: MyPy configuration integration
# ---------------------------------------------------------------------------


class TestMyPyConfiguration(unittest.TestCase):
    """Verify mypy.ini is correctly configured for stub discovery."""

    def test_mypy_ini_has_mypy_path(self) -> None:
        """mypy.ini must include mypy_path pointing to the stubs directory."""
        mypy_ini_path = os.path.join(_ROOT, "mypy.ini")
        with open(mypy_ini_path) as f:
            content = f.read()
        self.assertIn(
            "mypy_path",
            content,
            "mypy.ini must contain 'mypy_path' setting",
        )
        self.assertIn(
            "static/client/typing",
            content,
            "mypy_path must reference 'static/client/typing'",
        )

    def test_mypy_ini_has_test_file(self) -> None:
        """mypy.ini files list must include this test file."""
        mypy_ini_path = os.path.join(_ROOT, "mypy.ini")
        with open(mypy_ini_path) as f:
            content = f.read()
        self.assertIn(
            "server_tests/test_browser_typing_stubs.py",
            content,
            "mypy.ini files list must include test_browser_typing_stubs.py",
        )


if __name__ == "__main__":
    unittest.main()
