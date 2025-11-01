import unittest
from markdown_parser import MarkdownParser


class TestMarkdownParser(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = MarkdownParser()

    def test_headers(self) -> None:
        """Test all header levels H1-H6."""
        # Test H1
        result = self.parser.parse("# Header 1")
        self.assertIn("<h1>Header 1</h1>", result)
        
        # Test H2
        result = self.parser.parse("## Header 2")
        self.assertIn("<h2>Header 2</h2>", result)
        
        # Test H3
        result = self.parser.parse("### Header 3")
        self.assertIn("<h3>Header 3</h3>", result)
        
        # Test H4
        result = self.parser.parse("#### Header 4")
        self.assertIn("<h4>Header 4</h4>", result)
        
        # Test H5
        result = self.parser.parse("##### Header 5")
        self.assertIn("<h5>Header 5</h5>", result)
        
        # Test H6
        result = self.parser.parse("###### Header 6")
        self.assertIn("<h6>Header 6</h6>", result)

    def test_bold_text(self) -> None:
        """Test bold text formatting with both ** and __ syntax."""
        # Test ** syntax
        result = self.parser.parse("This is **bold** text")
        self.assertIn("<strong>bold</strong>", result)
        
        # Test __ syntax
        result = self.parser.parse("This is __bold__ text")
        self.assertIn("<strong>bold</strong>", result)
        
        # Test multiple bold in one line
        result = self.parser.parse("**First** and __second__ bold")
        self.assertIn("<strong>First</strong>", result)
        self.assertIn("<strong>second</strong>", result)

    def test_italic_text(self) -> None:
        """Test italic text formatting with both * and _ syntax."""
        # Test * syntax
        result = self.parser.parse("This is *italic* text")
        self.assertIn("<em>italic</em>", result)
        
        # Test _ syntax
        result = self.parser.parse("This is _italic_ text")
        self.assertIn("<em>italic</em>", result)
        
        # Test multiple italic in one line
        result = self.parser.parse("*First* and _second_ italic")
        self.assertIn("<em>First</em>", result)
        self.assertIn("<em>second</em>", result)

    def test_strikethrough_text(self) -> None:
        """Test strikethrough text formatting."""
        result = self.parser.parse("This is ~~strikethrough~~ text")
        self.assertIn("<del>strikethrough</del>", result)

    def test_inline_code(self) -> None:
        """Test inline code formatting."""
        result = self.parser.parse("This is `inline code` text")
        self.assertIn("<code>inline code</code>", result)

    def test_code_blocks(self) -> None:
        """Test code block formatting."""
        code_block = """```python
def hello():
    print("Hello, World!")
```"""
        result = self.parser.parse(code_block)
        self.assertIn("<pre><code>", result)
        self.assertIn("def hello():", result)
        self.assertIn("</code></pre>", result)

    def test_links(self) -> None:
        """Test link formatting."""
        result = self.parser.parse("This is a [link](https://example.com) text")
        self.assertIn('<a href="https://example.com">link</a>', result)

    def test_unordered_lists(self) -> None:
        """Test unordered list formatting."""
        markdown = """- Item 1
- Item 2
- Item 3"""
        result = self.parser.parse(markdown)
        self.assertIn("<ul>", result)
        self.assertIn("<li>Item 1</li>", result)
        self.assertIn("<li>Item 2</li>", result)
        self.assertIn("<li>Item 3</li>", result)
        self.assertIn("</ul>", result)

    def test_ordered_lists(self) -> None:
        """Test ordered list formatting."""
        markdown = """1. First item
2. Second item
3. Third item"""
        result = self.parser.parse(markdown)
        self.assertIn("<ol>", result)
        self.assertIn("<li>First item</li>", result)
        self.assertIn("<li>Second item</li>", result)
        self.assertIn("<li>Third item</li>", result)
        self.assertIn("</ol>", result)

    def test_nested_lists(self) -> None:
        """Test nested list formatting."""
        markdown = """- Item 1
  - Nested item 1
  - Nested item 2
- Item 2"""
        result = self.parser.parse(markdown)
        # Should contain nested ul structure
        self.assertIn("<ul>", result)
        self.assertIn("<li>Item 1</li>", result)
        self.assertIn("<li>Nested item 1</li>", result)
        self.assertIn("<li>Nested item 2</li>", result)
        self.assertIn("<li>Item 2</li>", result)

    def test_checkbox_lists(self) -> None:
        """Test checkbox list formatting."""
        markdown = """- [x] Completed task
- [ ] Incomplete task
- [x] Another completed task"""
        result = self.parser.parse(markdown)
        # Check for checkbox elements
        self.assertIn('class="checkbox checked"', result)
        self.assertIn('class="checkbox unchecked"', result)
        self.assertIn('class="checkbox-item"', result)
        self.assertIn("Completed task", result)
        self.assertIn("Incomplete task", result)

    def test_tables(self) -> None:
        """Test table formatting."""
        markdown = """| Name | Age | City |
|------|-----|------|
| John | 25  | NYC  |
| Jane | 30  | LA   |"""
        result = self.parser.parse(markdown)
        self.assertIn("<table>", result)
        self.assertIn("<th>Name</th>", result)
        self.assertIn("<th>Age</th>", result)
        self.assertIn("<th>City</th>", result)
        self.assertIn("<td>John</td>", result)
        self.assertIn("<td>25</td>", result)
        self.assertIn("<td>NYC</td>", result)
        self.assertIn("</table>", result)

    def test_blockquotes(self) -> None:
        """Test blockquote formatting."""
        result = self.parser.parse("> This is a blockquote")
        self.assertIn("<blockquote>This is a blockquote</blockquote>", result)

    def test_horizontal_rules(self) -> None:
        """Test horizontal rule formatting."""
        result = self.parser.parse("---")
        self.assertIn("<hr>", result)

    def test_inline_math_expressions(self) -> None:
        """Test inline mathematical expressions."""
        result = self.parser.parse("The equation \\(E = mc^2\\) is famous")
        self.assertIn('class="math-inline"', result)
        self.assertIn("\\(E = mc^2\\)", result)

    def test_block_math_expressions(self) -> None:
        """Test block mathematical expressions."""
        markdown = """$$
\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}
$$"""
        result = self.parser.parse(markdown)
        self.assertIn('class="math-block"', result)
        self.assertIn("\\int_{0}^{\\infty}", result)

    def test_mixed_formatting(self) -> None:
        """Test multiple formatting elements together."""
        markdown = """# Main Title

This is **bold** and *italic* text with `inline code`.

## Math Section

The famous equation is \\(E = mc^2\\).

### List Example

- Item with **bold** text
- Item with *italic* text
- [Link item](https://example.com)

| Formula | Result |
|---------|--------|
| \\(2 + 2\\) | 4 |
| \\(x^2\\) | Variable |"""
        
        result = self.parser.parse(markdown)
        
        # Check headers
        self.assertIn("<h1>Main Title</h1>", result)
        self.assertIn("<h2>Math Section</h2>", result)
        self.assertIn("<h3>List Example</h3>", result)
        
        # Check formatting
        self.assertIn("<strong>bold</strong>", result)
        self.assertIn("<em>italic</em>", result)
        self.assertIn("<code>inline code</code>", result)
        
        # Check math
        self.assertIn('class="math-inline"', result)
        
        # Check list
        self.assertIn("<ul>", result)
        self.assertIn("<li>Item with", result)
        
        # Check table
        self.assertIn("<table>", result)
        self.assertIn("<th>Formula</th>", result)

    def test_edge_cases(self) -> None:
        """Test edge cases and malformed markdown."""
        # Empty string
        result = self.parser.parse("")
        self.assertEqual(result, "")
        
        # Only whitespace
        result = self.parser.parse("   \n  \n  ")
        # Should handle gracefully
        
        # Unmatched formatting
        result = self.parser.parse("**bold without closing")
        # Should not crash
        
        # Nested formatting
        result = self.parser.parse("**bold with *italic* inside**")
        # Should handle reasonably

    def test_multiple_math_expressions(self) -> None:
        """Test multiple math expressions in one text."""
        markdown = "First equation \\(x = 1\\) and second $$y = 2$$ then \\(z = 3\\)"
        result = self.parser.parse(markdown)
        
        # Should contain multiple math spans/divs
        math_inline_count = result.count('class="math-inline"')
        math_block_count = result.count('class="math-block"')
        
        self.assertEqual(math_inline_count, 2)  # Two inline expressions
        self.assertEqual(math_block_count, 1)   # One block expression

    def test_table_with_inline_formatting(self) -> None:
        """Test tables with inline formatting in cells."""
        markdown = """| **Bold** | *Italic* | `Code` |
|----------|----------|--------|
| **Data** | *Info*   | `var`  |"""
        
        result = self.parser.parse(markdown)
        self.assertIn("<strong>Bold</strong>", result)
        self.assertIn("<em>Italic</em>", result)
        self.assertIn("<code>Code</code>", result)

    def test_list_with_mixed_types(self) -> None:
        """Test mixing ordered and unordered lists."""
        markdown = """1. Ordered item 1
2. Ordered item 2

- Unordered item 1
- Unordered item 2"""
        
        result = self.parser.parse(markdown)
        self.assertIn("<ol>", result)
        self.assertIn("</ol>", result)
        self.assertIn("<ul>", result)
        self.assertIn("</ul>", result)

    def test_complex_nested_structure(self) -> None:
        """Test complex nested markdown structure."""
        markdown = """## Complex Example

This has:

1. **Bold** ordered item
   - *Italic* nested item
   - Another nested with `code`
2. Second ordered item

### Math Examples

Inline: \\(f(x) = x^2\\)

Block:
$$
\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}
$$

> Quote with **bold** text

| Function | Derivative |
|----------|------------|
| \\(x^2\\) | \\(2x\\) |
| \\(\\sin x\\) | \\(\\cos x\\) |"""
        
        result = self.parser.parse(markdown)
        
        # Verify all elements are present
        self.assertIn("<h2>Complex Example</h2>", result)
        self.assertIn("<h3>Math Examples</h3>", result)
        self.assertIn("<ol>", result)
        self.assertIn("<ul>", result)
        self.assertIn("<blockquote>", result)
        self.assertIn("<table>", result)
        self.assertIn('class="math-inline"', result)
        self.assertIn('class="math-block"', result)
        self.assertIn("<strong>Bold</strong>", result)
        self.assertIn("<em>Italic</em>", result)
        self.assertIn("<code>code</code>", result)

    def test_underscore_in_words_not_italic(self) -> None:
        """Test that underscores in the middle of words are not treated as italic formatting."""
        # Test cases where underscores should NOT be italicized
        test_cases = [
            ("some_variable_name", "some_variable_name"),  # Variable names
            ("file_name.txt", "file_name.txt"),  # File names
            ("hello_world_function", "hello_world_function"),  # Function names
            ("test_case_one", "test_case_one"),  # General underscores in words
            ("multiple_under_scores_here", "multiple_under_scores_here"),  # Multiple underscores
            ("start_with_underscore", "start_with_underscore"),  # Starts with underscore case
            ("end_with_underscore_", "end_with_underscore_"),  # Ends with underscore
        ]
        
        for input_text, expected_content in test_cases:
            with self.subTest(input_text=input_text):
                result = self.parser.parse(input_text)
                # Should not contain <em> tags
                self.assertNotIn("<em>", result, f"Underscore in '{input_text}' should not be italicized")
                # Should contain the original text
                self.assertIn(expected_content, result, f"Original text should be preserved in '{input_text}'")

    def test_underscore_with_spaces_is_italic(self) -> None:
        """Test that underscores surrounded by spaces are properly italicized."""
        # Test cases where underscores SHOULD be italicized
        test_cases = [
            ("This is _italic_ text", "<em>italic</em>"),
            ("Start _italic_ middle _another_ end", ["<em>italic</em>", "<em>another</em>"]),
            ("Just _one word_ here", "<em>one word</em>"),
            ("At the _beginning_ of sentence", "<em>beginning</em>"),
            ("At the end of _sentence_", "<em>sentence</em>"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.parser.parse(input_text)
                if isinstance(expected, list):
                    for exp in expected:
                        self.assertIn(exp, result, f"Expected '{exp}' in result for '{input_text}'")
                else:
                    self.assertIn(expected, result, f"Expected '{expected}' in result for '{input_text}'")

    def test_mixed_underscore_scenarios(self) -> None:
        """Test mixed scenarios with both underscore types."""
        # Test cases with both italics and non-italics underscores
        test_cases = [
            ("This has _italic text_ and some_variable_name", ["<em>italic text</em>", "some_variable_name"]),
            ("function_name() and _emphasis_ in same line", ["function_name()", "<em>emphasis</em>"]),
            ("file_name.txt contains _important_ data", ["file_name.txt", "<em>important</em>"]),
            ("variable_one and variable_two with _italic_", ["variable_one", "variable_two", "<em>italic</em>"]),
        ]
        
        for input_text, expected_items in test_cases:
            with self.subTest(input_text=input_text):
                result = self.parser.parse(input_text)
                for item in expected_items:
                    self.assertIn(item, result, f"Expected '{item}' in result for '{input_text}'")

    def test_double_underscore_bold(self) -> None:
        """Test that double underscores with proper spacing create bold text."""
        # Test cases where double underscores SHOULD be bold
        test_cases = [
            ("This is __bold__ text", "<strong>bold</strong>"),
            ("Start __bold__ middle __another__ end", ["<strong>bold</strong>", "<strong>another</strong>"]),
            ("Just __one word__ here", "<strong>one word</strong>"),
            ("At the __beginning__ of sentence", "<strong>beginning</strong>"),
            ("At the end of __sentence__", "<strong>sentence</strong>"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.parser.parse(input_text)
                if isinstance(expected, list):
                    for exp in expected:
                        self.assertIn(exp, result, f"Expected '{exp}' in result for '{input_text}'")
                else:
                    self.assertIn(expected, result, f"Expected '{expected}' in result for '{input_text}'")

    def test_malformed_underscore_patterns(self) -> None:
        """Test that malformed underscore patterns are not parsed as formatting."""
        # Test cases that should NOT be parsed as formatting
        test_cases = [
            ("_something", "_something"),  # Unclosed single underscore
            ("something_", "something_"),  # Trailing single underscore
            ("__something", "__something"),  # Unclosed double underscore
            ("something__", "something__"),  # Trailing double underscore
            ("__something__else", "__something__else"),  # Double underscore not properly spaced
            ("text__bold__more", "text__bold__more"),  # Double underscore attached to text
            ("_italic_more", "_italic_more"),  # Single underscore attached to text
            ("text_italic_", "text_italic_"),  # Single underscore attached to text
            ("pre__bold__post", "pre__bold__post"),  # Double underscore without spaces
            ("pre_italic_post", "pre_italic_post"),  # Single underscore without spaces
        ]
        
        for input_text, expected_content in test_cases:
            with self.subTest(input_text=input_text):
                result = self.parser.parse(input_text)
                # Should not contain formatting tags
                self.assertNotIn("<em>", result, f"Malformed underscore in '{input_text}' should not be italicized")
                self.assertNotIn("<strong>", result, f"Malformed underscore in '{input_text}' should not be bold")
                # Should contain the original text
                self.assertIn(expected_content, result, f"Original text should be preserved in '{input_text}'")

    def test_underscore_edge_cases_with_punctuation(self) -> None:
        """Test underscore behavior with punctuation and special characters."""
        # Test cases with punctuation that SHOULD work
        punctuation_cases = [
            ("Hello, _italic_ text!", "<em>italic</em>"),
            ("Start. __Bold__ text.", "<strong>Bold</strong>"),
            ("Question _italic_?", "<em>italic</em>"),
            ("Exclamation __bold__!", "<strong>bold</strong>"),
            ("Parentheses (_italic_)", "<em>italic</em>"),
            ("Brackets [__bold__]", "<strong>bold</strong>"),
        ]
        
        for input_text, expected in punctuation_cases:
            with self.subTest(input_text=input_text):
                result = self.parser.parse(input_text)
                self.assertIn(expected, result, f"Expected '{expected}' in result for '{input_text}'")

    def test_mixed_formatting_with_underscores(self) -> None:
        """Test mixed formatting scenarios with underscores."""
        markdown = "This has _italic_ and some_variable_name and **bold_text** formatting."
        result = self.parser.parse(markdown)
        
        # Should italicize spaced underscores but not variable names
        self.assertIn("<em>italic</em>", result)
        self.assertIn("some_variable_name", result)  # Should remain unchanged
        self.assertIn("<strong>bold_text</strong>", result)  # Bold should work

    def test_mathematical_expressions_not_tables(self) -> None:
        """Test that mathematical expressions with pipes are NOT parsed as tables."""
        # Mathematical expressions that should NOT become tables
        math_examples = [
            "The absolute value |x| is always non-negative.",
            "Consider the set {x | x > 0}.",
            "The expression |x + y| = |x| + |y| when x, y ≥ 0.",
            "Find all x such that |x - 2| < 5.",
            "In mathematics, we often use |A| to denote the cardinality of set A.",
            "The modulus operation: 7 mod 3 = 1, written as 7 | 3.",
            "Vector notation: |v| represents the magnitude of vector v.",
            "Set builder notation: {x ∈ ℝ | x² > 4}.",
            "Conditional probability: P(A | B) = P(A ∩ B) / P(B).",
            "Matrix determinant: |A| for matrix A."
        ]
        
        for example in math_examples:
            with self.subTest(math_expression=example):
                result = self.parser.parse(example)
                # Should NOT contain table tags
                self.assertNotIn("<table>", result, f"Math expression '{example}' was incorrectly parsed as table")
                self.assertNotIn("<th", result, f"Math expression '{example}' contains table headers")
                self.assertNotIn("<td", result, f"Math expression '{example}' contains table cells")
                # Should contain the original pipe characters
                self.assertIn("|", result, f"Original pipe characters missing from '{example}'")

    def test_proper_markdown_tables(self) -> None:
        """Test that proper markdown tables ARE correctly parsed."""
        # Basic table
        basic_table = """| Name | Age | City |
|------|-----|------|
| John | 30  | NYC  |
| Jane | 25  | LA   |"""
        
        result = self.parser.parse(basic_table)
        self.assertIn("<table>", result)
        self.assertIn("<thead>", result)
        self.assertIn("<tbody>", result)
        self.assertIn("<th>Name</th>", result)
        self.assertIn("<th>Age</th>", result)
        self.assertIn("<th>City</th>", result)
        self.assertIn("<td>John</td>", result)
        self.assertIn("<td>30</td>", result)
        self.assertIn("<td>NYC</td>", result)

    def test_table_alignment(self) -> None:
        """Test table column alignment parsing."""
        aligned_table = """| Left | Center | Right |
|:-----|:------:|------:|
| A    | B      | C     |
| 1    | 2      | 3     |"""
        
        result = self.parser.parse(aligned_table)
        self.assertIn("<table>", result)
        self.assertIn('text-align: left;', result)
        self.assertIn('text-align: center;', result)
        self.assertIn('text-align: right;', result)

    def test_table_without_separator_not_table(self) -> None:
        """Test that lines with pipes but no separator row are NOT tables."""
        not_tables = [
            "| This has pipes | but no separator |",
            "| Single line | with pipes |",
            """| First line |
| Second line |""",  # No separator between them
            "Just text | with pipes | scattered around",
            "| Start pipe only",
            "End pipe only |",
            "Middle | pipe | here"
        ]
        
        for example in not_tables:
            with self.subTest(not_table=example):
                result = self.parser.parse(example)
                self.assertNotIn("<table>", result, f"Non-table '{example}' was incorrectly parsed as table")

    def test_malformed_table_separators(self) -> None:
        """Test that malformed separator rows don't create tables."""
        malformed_tables = [
            """| Header |
| Not a separator |
| Data |""",  # Second line is not a valid separator
            
            """| Header |
|====|
| Data |""",  # Equals instead of hyphens
            
            """| Header |
| abc |
| Data |""",  # Letters instead of hyphens
            
            """| Header |
|  |
| Data |""",  # Empty separator
        ]
        
        for example in malformed_tables:
            with self.subTest(malformed=example):
                result = self.parser.parse(example)
                self.assertNotIn("<table>", result, f"Malformed table was incorrectly parsed: {example}")

    def test_valid_table_separators(self) -> None:
        """Test various valid separator row formats."""
        valid_separators = [
            """| Header |
|--------|
| Data |""",  # Basic separator
            
            """| Left | Right |
|:-----|------:|
| A    | B     |""",  # With alignment
            
            """| A | B | C |
|---|:-:|--:|
| 1 | 2 | 3 |""",  # Mixed alignment
            
            """| Header |
| --- |
| Data |""",  # Minimal separator
            
            """|Header|
|---|
|Data|""",  # No spaces around pipes
        ]
        
        for example in valid_separators:
            with self.subTest(valid_table=example):
                result = self.parser.parse(example)
                self.assertIn("<table>", result, f"Valid table was not parsed: {example}")
                self.assertIn("<th", result)  # Match <th with or without attributes
                self.assertIn("<td", result)  # Match <td with or without attributes

    def test_table_with_math_expressions_in_cells(self) -> None:
        """Test tables that contain mathematical expressions within cells."""
        table_with_math = """| Expression | Value |
|------------|-------|
| \\|x\\|     | abs(x) |
| {x \\| x > 0} | positive reals |
| \\(E = mc^2\\) | Einstein |"""
        
        result = self.parser.parse(table_with_math)
        self.assertIn("<table>", result)
        self.assertIn("<th>Expression</th>", result)
        self.assertIn("<th>Value</th>", result)
        # Math expressions should be preserved in cells
        self.assertIn("abs(x)", result)
        self.assertIn("positive reals", result)
        self.assertIn("Einstein", result)

    def test_mixed_content_with_tables_and_math(self) -> None:
        """Test content mixing tables and mathematical expressions."""
        mixed_content = """# Math and Tables

The absolute value |x| is important.

| Function | Definition |
|----------|------------|
| |x|      | absolute value |
| {x \\| x > 0} | positive set |

Set notation {y \\| y < 0} represents negative numbers.

Another table:

| A | B |
|---|---|
| 1 | 2 |

Final math: |a + b| ≤ |a| + |b|."""
        
        result = self.parser.parse(mixed_content)
        
        # Should have tables
        table_count = result.count("<table>")
        self.assertEqual(table_count, 2, "Should have exactly 2 tables")
        
        # Should have math expressions that are NOT in tables
        self.assertIn("The absolute value |x| is important", result)
        self.assertIn("Set notation {y", result)
        self.assertIn("Final math: |a + b|", result)
        
        # Table content should be present
        self.assertIn("<th>Function</th>", result)
        self.assertIn("<th>Definition</th>", result)

    def test_lines_not_starting_with_pipe_not_tables(self) -> None:
        """Test that lines not starting with pipes are never parsed as tables."""
        # These lines have pipes but don't start with pipes (after whitespace)
        # and should NEVER be considered tables, even with separators
        non_pipe_starting_lines = [
            # Lines with pipes but not starting with pipes
            "Text with | pipes | in middle",
            "Some content | and more | content here",
            "  Text with leading spaces | and pipes | but no leading pipe",
            "Math expression |x| in sentence",
            "Set notation {x | x > 0} explained",
            "Conditional probability P(A | B) formula",
            
            # Even with what looks like separator lines after
            """Text with | pipes | in middle
|-----|-----|
More text | here | too""",
            
            """Math |x| and |y| values
|---|---|
Not a table | still | not""",
            
            # Mixed content
            """Regular text with | pipes | scattered
| Header | Column |
|--------|--------|
| Data   | Value  |
More text | with | pipes""",
        ]
        
        for example in non_pipe_starting_lines:
            with self.subTest(non_pipe_line=example):
                result = self.parser.parse(example)
                # Count tables - should be 0 for most, or 1 only if there's a valid table section
                table_count = result.count("<table>")
                
                # For single line examples, should be 0
                if '\n' not in example:
                    self.assertEqual(table_count, 0, 
                                   f"Single line not starting with pipe was parsed as table: '{example}'")
                # For multi-line examples, check that lines not starting with pipes don't create tables
                else:
                    lines = example.split('\n')
                    valid_table_lines = [line for line in lines if line.strip().startswith('|')]
                    non_table_lines = [line for line in lines if not line.strip().startswith('|')]
                    
                    # Non-table lines should not be in table HTML
                    for non_table_line in non_table_lines:
                        if '|' in non_table_line:
                            # Make sure this text appears outside of table tags
                            pipe_content = non_table_line.split('|')[0].strip()
                            if pipe_content:
                                # This content should appear in the result but not inside table tags
                                self.assertIn(pipe_content, result, 
                                            f"Content '{pipe_content}' should appear in result")

    def test_edge_cases_pipes_and_tables(self) -> None:
        """Test edge cases with pipes and potential table confusion."""
        edge_cases = [
            # Single pipe in text
            ("Just a | pipe", False),
            
            # Pipes at start/end
            ("|Starting pipe", False),
            ("Ending pipe|", False),
            
            # Multiple pipes but no valid table structure
            ("| A | B | C |", False),  # No separator
            
            # Valid minimal table
            ("""| A |
|---|
| 1 |""", True),
            
            # Mathematical expressions
            ("Function f(x) = |x - 1| + |x + 1|", False),
            ("Probability P(A|B) = 0.5", False),
            
            # Empty table cells
            ("""| A | B |
|---|---|
|   |   |""", True),
        ]
        
        for content, should_be_table in edge_cases:
            with self.subTest(content=content, should_be_table=should_be_table):
                result = self.parser.parse(content)
                if should_be_table:
                    self.assertIn("<table>", result, f"Should be table: {content}")
                    self.assertIn("<th", result, f"Should have headers: {content}")
                    self.assertIn("<td", result, f"Should have cells: {content}")
                else:
                    self.assertNotIn("<table>", result, f"Should NOT be table: {content}")


if __name__ == '__main__':
    unittest.main()