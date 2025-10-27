"""
Markdown Parser for MatHud Chat Interface

A comprehensive markdown parser that supports:
- Headers (H1-H6: # ## ### #### ##### ######)
- Bold text (**bold** or __bold__)
- Italic text (*italic* or _italic_)
- Strikethrough text (~~strikethrough~~)
- Inline and block code (` and ```)
- Links and images
- Ordered and unordered lists with nesting
- Checkboxes (- [x] and - [ ])
- Tables
- Blockquotes
- Horizontal rules
- Mathematical expressions (LaTeX: \(...\) for inline, $$...$$ for block)
"""

import re

class MarkdownParser:
    """Custom markdown parser optimized for chat interface display."""
    
    def parse(self, text):
        """Parse markdown text to HTML."""
        try:
            # Skip Brython's apply_markdown as it's not working properly
            # It wraps everything in a single <p> tag and doesn't parse headers correctly
            html_content = self._simple_markdown_parse(text)
            return html_content
            
        except Exception as e:
            print(f"Error in custom markdown parsing: {e}")
            # Ultimate fallback
            return text.replace('\n', '<br>')

    def _simple_markdown_parse(self, text):
        """Simple markdown parser for basic formatting using string operations."""
        try:
            # First handle tables
            text = self._process_tables(text)
            
            # Split text into lines for processing
            lines = text.split('\n')
            html_lines = []
            in_code_block = False
            code_block_content = []
            
            for line in lines:
                # Skip table processing if already processed
                if '<table>' in line or '</table>' in line or '<tr>' in line or '<td>' in line or '<th>' in line:
                    html_lines.append(line)
                    continue
                    
                # Handle code blocks
                if line.strip().startswith('```'):
                    if in_code_block:
                        # End code block
                        code_content = '\n'.join(code_block_content)
                        html_lines.append(f'<pre><code>{code_content}</code></pre>')
                        code_block_content = []
                        in_code_block = False
                    else:
                        # Start code block
                        in_code_block = True
                    continue
                
                if in_code_block:
                    code_block_content.append(line)
                    continue
                
                # Process other markdown elements
                processed_line = line

                heading_match = self._parse_heading(processed_line)
                if heading_match:
                    level, heading_content = heading_match
                    processed_line = f'<h{level}>{heading_content}</h{level}>'
                # Lists - handle ordered and unordered with indentation
                elif self._is_list_item(processed_line):
                    processed_line = self._process_list_item(processed_line)
                # Blockquotes
                elif processed_line.startswith('> '):
                    processed_line = f'<blockquote>{processed_line[2:]}</blockquote>'
                # Horizontal rules
                elif processed_line.strip() == '---':
                    processed_line = '<hr>'
                
                # Handle inline formatting
                processed_line = self._process_inline_markdown(processed_line)
                
                html_lines.append(processed_line)
            
            # Join lines and wrap list items
            html = self._join_lines_with_smart_breaks(html_lines)
            html = self._wrap_list_items_improved(html)
            
            # Process mathematical expressions after everything else
            html = self._process_math_expressions(html)
            
            return html
            
        except Exception as e:
            print(f"Error in simple markdown parsing: {e}")
            # Ultimate fallback
            return text.replace('\n', '<br>')

    def _parse_heading(self, line):
        """Parse markdown heading and return (level, content) if matched."""
        if not line:
            return None

        match = re.match(r"^(#{1,6})(?:\s+|\b)(.*)$", line)
        if not match:
            return None

        hashes, content = match.groups()
        level = len(hashes)

        content = content.lstrip()
        if not content:
            return (level, "")

        return (level, content)
    
    def _process_tables(self, text):
        """Process markdown tables using proper GFM table parsing algorithm."""
        lines = text.split('\n')
        result_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this could be start of a table - must start with pipe
            if line.strip().startswith('|') and line.strip():
                # Look ahead to see if next line is a delimiter
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if self._is_delimiter_row(next_line):
                        # Found table start - collect all table rows
                        table_lines = [line, next_line]
                        j = i + 2
                        
                        # Collect data rows - must also start with pipe
                        while j < len(lines):
                            if lines[j].strip().startswith('|') and lines[j].strip():
                                table_lines.append(lines[j])
                                j += 1
                            else:
                                break
                        
                        # Process the table
                        table_html = self._build_table_html(table_lines)
                        result_lines.append(table_html)
                        i = j
                        continue
            
            # Not a table - add line as is
            result_lines.append(line)
            i += 1
        
        return '\n'.join(result_lines)
    
    def _build_table_html(self, table_lines):
        """Build HTML table from table lines."""
        if len(table_lines) < 2:
            return '\n'.join(table_lines)
        
        header_line = table_lines[0]
        delimiter_line = table_lines[1]
        data_lines = table_lines[2:] if len(table_lines) > 2 else []
        
        # Parse header
        header_cells = self._parse_table_row(header_line)
        if not header_cells:
            return '\n'.join(table_lines)
        
        # Parse alignments
        alignments = self._parse_alignments(delimiter_line)
        
        # Build HTML - no internal newlines to avoid extra spacing
        html = '<table><thead><tr>'
        for i, cell in enumerate(header_cells):
            align = alignments[i] if i < len(alignments) else ''
            align_attr = f' style="text-align: {align};"' if align else ''
            html += f'<th{align_attr}>{cell}</th>'
        html += '</tr></thead>'
        
        # Process data rows
        if data_lines:
            html += '<tbody>'
            for data_line in data_lines:
                data_cells = self._parse_table_row(data_line)
                if data_cells:
                    html += '<tr>'
                    for i, cell in enumerate(data_cells):
                        align = alignments[i] if i < len(alignments) else ''
                        align_attr = f' style="text-align: {align};"' if align else ''
                        html += f'<td{align_attr}>{cell}</td>'
                    html += '</tr>'
            html += '</tbody>'
        
        html += '</table>'
        return html
    
    def _is_delimiter_row(self, line):
        """Check if a line is a valid table delimiter row."""
        # Delimiter row must start with | (after whitespace)
        line = line.strip()
        if not line.startswith('|'):
            return False
            
        # Remove leading/trailing pipes
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]
        
        # Split by pipes and check each cell
        cells = [cell.strip() for cell in line.split('|')]
        
        # Must have at least one valid delimiter cell
        valid_cell_count = 0
        for cell in cells:
            if not cell:
                # Empty cells are allowed but don't count toward validity
                continue
            # Must be hyphens with optional colons for alignment
            if re.match(r'^:?-+:?$', cell):
                valid_cell_count += 1
            else:
                return False  # Invalid delimiter character
        
        # Must have at least one valid delimiter cell
        return valid_cell_count > 0
    
    def _parse_table_row(self, line):
        """Parse a table row and return cell contents."""
        # Remove leading/trailing whitespace
        line = line.strip()
        
        # Remove leading/trailing pipes if present
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]
        
        # Split by pipes and process each cell
        cells = [cell.strip() for cell in line.split('|')]
        
        # Process inline markdown in each cell
        processed_cells = []
        for cell in cells:
            if cell:
                # Process inline markdown (bold, italic, etc.)
                processed_cell = self._process_inline_markdown(cell)
                processed_cells.append(processed_cell)
            else:
                processed_cells.append('')
        
        return processed_cells
    
    def _parse_alignments(self, delimiter_line):
        """Parse column alignments from delimiter row."""
        # Remove leading/trailing whitespace and optional pipes
        line = delimiter_line.strip()
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]
        
        # Split by pipes and determine alignment for each column
        cells = [cell.strip() for cell in line.split('|')]
        alignments = []
        
        for cell in cells:
            if not cell:
                alignments.append('')
                continue
                
            starts_with_colon = cell.startswith(':')
            ends_with_colon = cell.endswith(':')
            
            if starts_with_colon and ends_with_colon:
                alignments.append('center')
            elif starts_with_colon:
                alignments.append('left')
            elif ends_with_colon:
                alignments.append('right')
            else:
                alignments.append('')
        
        return alignments
    
    def _is_list_item(self, line):
        """Check if a line is a list item (ordered, unordered, or checkbox)."""
        stripped = line.strip()
        
        # Checkbox items
        if stripped.startswith('- [x]') or stripped.startswith('- [ ]'):
            return True
            
        # Unordered list
        if stripped.startswith('- ') or stripped.startswith('* '):
            return True
            
        # Ordered list (number followed by period and space)
        if len(stripped) > 2 and stripped[1:3] == '. ':
            try:
                int(stripped[0])  # Check if first char is number
                return True
            except:
                pass
                
        # Handle multi-digit numbers
        parts = stripped.split('. ', 1)
        if len(parts) == 2:
            try:
                int(parts[0])
                return True
            except:
                pass
        return False
    
    def _process_list_item(self, line):
        """Process a list item and determine its type and indentation."""
        # Count leading spaces for indentation level
        leading_spaces = len(line) - len(line.lstrip())
        indent_level = leading_spaces // 2  # 2 spaces = 1 indent level
        
        stripped = line.strip()
        
        # Handle checkboxes
        if stripped.startswith('- [x]'):
            content = stripped[6:]  # Remove '- [x] '
            checkbox = '<span class="checkbox checked">✓</span>'
            return f'<li class="checkbox-item" data-list-type="ul" data-indent="{indent_level}">{checkbox}{content}</li>'
        elif stripped.startswith('- [ ]'):
            content = stripped[6:]  # Remove '- [ ] '
            checkbox = '<span class="checkbox unchecked"></span>'
            return f'<li class="checkbox-item" data-list-type="ul" data-indent="{indent_level}">{checkbox}{content}</li>'
        
        # Determine list type and content
        if stripped.startswith('- ') or stripped.startswith('* '):
            # Unordered list
            content = stripped[2:]
            list_type = 'ul'
        else:
            # Ordered list (number followed by period)
            parts = stripped.split('. ', 1)
            if len(parts) == 2:
                try:
                    int(parts[0])
                    content = parts[1]
                    list_type = 'ol'
                except:
                    # Fallback
                    content = stripped
                    list_type = 'ul'
            else:
                content = stripped
                list_type = 'ul'
        
        # Add data attributes to track list type and indent level
        return f'<li data-list-type="{list_type}" data-indent="{indent_level}">{content}</li>'
    
    def _join_lines_with_smart_breaks(self, lines):
        """Join lines with smart line break handling."""
        try:
            result_lines = []
            
            for i, line in enumerate(lines):
                if line.strip():  # Non-empty line
                    result_lines.append(line)
                else:
                    # Empty line - only add break if not between list items or table elements
                    if i > 0 and i < len(lines) - 1:
                        prev_line = lines[i-1].strip()
                        next_line = lines[i+1].strip()
                        
                        # Don't add breaks between list items
                        prev_is_list = '<li' in prev_line
                        next_is_list = '<li' in next_line
                        
                        # Don't add breaks around tables
                        prev_is_table = any(tag in prev_line for tag in ['<table>', '</table>', '<tr>', '<td>', '<th>'])
                        next_is_table = any(tag in next_line for tag in ['<table>', '</table>', '<tr>', '<td>', '<th>'])
                        
                        if not (prev_is_list and next_is_list) and not (prev_is_table or next_is_table):
                            result_lines.append('<br>')
            
            return '<br>'.join(result_lines)
            
        except Exception as e:
            print(f"Error joining lines: {e}")
            return '<br>'.join(lines)

    def _wrap_list_items_improved(self, html):
        """Wrap list items with proper <ul>/<ol> tags and handle REAL nesting."""
        try:
            lines = html.split('<br>')
            result = []
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                if '<li data-list-type=' in line and '</li>' in line:
                    # Start processing a list
                    list_items = []
                    current_index = i
                    
                    # Collect all consecutive list items
                    while current_index < len(lines):
                        current_line = lines[current_index].strip()
                        if '<li data-list-type=' in current_line and '</li>' in current_line:
                            list_type = self._extract_data_attr(current_line, 'data-list-type')
                            indent_level = int(self._extract_data_attr(current_line, 'data-indent') or '0')
                            
                            # Clean the line (remove data attributes)
                            clean_line = current_line.replace(f' data-list-type="{list_type}"', '')
                            clean_line = clean_line.replace(f' data-indent="{indent_level}"', '')
                            
                            list_items.append((clean_line, list_type, indent_level))
                            current_index += 1
                        else:
                            break
                    
                    # Process the collected list items into nested HTML
                    nested_html = self._build_nested_list_html(list_items)
                    result.append(nested_html)
                    i = current_index
                else:
                    # Not a list item
                    if line:
                        result.append(line)
                    i += 1
            
            # Join with line breaks
            return '<br>'.join(result)
            
        except Exception as e:
            print(f"Error in improved list wrapping: {e}")
            return html
    
    def _extract_data_attr(self, line, attr_name):
        """Extract data attribute value from HTML line."""
        try:
            start = line.find(f'{attr_name}="') + len(f'{attr_name}="')
            end = line.find('"', start)
            return line[start:end] if start > len(f'{attr_name}="') - 1 and end > start else None
        except:
            return None
    
    def _build_nested_list_html(self, list_items):
        """Build properly nested HTML from list items."""
        try:
            if not list_items:
                return ''
            
            result = []
            stack = []  # Stack of (list_type, indent_level)
            
            for item_html, list_type, indent_level in list_items:
                # Close lists that are deeper than current level
                while stack and stack[-1][1] >= indent_level:
                    if stack[-1][1] == indent_level and stack[-1][0] == list_type:
                        # Same level and type, continue
                        break
                    # Close the deeper list
                    closed_type = stack.pop()[0]
                    tag = 'ol' if closed_type == 'ol' else 'ul'
                    result.append(f'</{tag}>')
                
                # Open new list if needed
                if not stack or stack[-1][1] < indent_level or stack[-1][0] != list_type:
                    tag = 'ol' if list_type == 'ol' else 'ul'
                    result.append(f'<{tag}>')
                    stack.append((list_type, indent_level))
                
                # Add the list item
                result.append(item_html)
            
            # Close all remaining open lists
            while stack:
                closed_type = stack.pop()[0]
                tag = 'ol' if closed_type == 'ol' else 'ul'
                result.append(f'</{tag}>')
            
            return ''.join(result)
            
        except Exception as e:
            print(f"Error building nested list HTML: {e}")
            return ''
    
    def _process_inline_markdown(self, text):
        """Process inline markdown elements like bold, italic, code."""
        try:
            # Bold text (**text** and __text__)
            while '**' in text:
                start = text.find('**')
                if start == -1:
                    break
                end = text.find('**', start + 2)
                if end == -1:
                    break
                before = text[:start]
                content = text[start + 2:end]
                after = text[end + 2:]
                text = before + f'<strong>{content}</strong>' + after
            
            # Process double underscores for bold - restart search after each replacement
            while '__' in text:
                found_match = False
                pos = 0
                while pos < len(text):
                    start = text.find('__', pos)
                    if start == -1:
                        break
                    end = text.find('__', start + 2)
                    if end == -1:
                        break
                    
                    # Check if double underscore is surrounded by proper word boundaries
                    char_before = text[start - 1] if start > 0 else ' '
                    char_after = text[end + 2] if end + 2 < len(text) else ' '
                    
                    # Only apply bold formatting if both double underscores are at proper word boundaries
                    # Must be preceded and followed by space, punctuation, or start/end of text
                    before_is_boundary = (char_before == ' ' or 
                                         char_before in '.,!?:;()[]{}"\'-' or 
                                         start == 0)
                    after_is_boundary = (char_after == ' ' or 
                                        char_after in '.,!?:;()[]{}"\'-' or 
                                        end + 2 >= len(text))
                    
                    if before_is_boundary and after_is_boundary:
                        before = text[:start]
                        content = text[start + 2:end]
                        after = text[end + 2:]
                        text = before + f'<strong>{content}</strong>' + after
                        found_match = True
                        break  # Break inner loop to restart search from beginning
                    else:
                        # Skip this pair and continue searching
                        pos = start + 2
                
                # If no match found in this pass, exit the outer loop
                if not found_match:
                    break
            
            # Italic text (*text* and _text_)
            while '*' in text:  # Process asterisks regardless of bold tags
                start = text.find('*')
                if start == -1:
                    break
                end = text.find('*', start + 1)
                if end == -1:
                    break
                before = text[:start]
                content = text[start + 1:end]
                after = text[end + 1:]
                text = before + f'<em>{content}</em>' + after
            
            # Enhanced underscore italic processing - only format when surrounded by spaces or at word boundaries
            while '_' in text:  # Process single underscores regardless of bold tags
                found_match = False
                pos = 0
                while pos < len(text):
                    start = text.find('_', pos)
                    if start == -1:
                        break
                    end = text.find('_', start + 1)
                    if end == -1:
                        break
                    
                    # Check if underscore is surrounded by proper word boundaries
                    char_before = text[start - 1] if start > 0 else ' '
                    char_after = text[end + 1] if end + 1 < len(text) else ' '
                    
                    # Only apply italic formatting if both underscores are at proper word boundaries
                    # Must be preceded and followed by space, punctuation, or start/end of text
                    before_is_boundary = (char_before == ' ' or 
                                         char_before in '.,!?:;()[]{}"\'-' or 
                                         start == 0)
                    after_is_boundary = (char_after == ' ' or 
                                        char_after in '.,!?:;()[]{}"\'-' or 
                                        end + 1 >= len(text))
                    
                    if before_is_boundary and after_is_boundary:
                        before = text[:start]
                        content = text[start + 1:end]
                        after = text[end + 1:]
                        text = before + f'<em>{content}</em>' + after
                        found_match = True
                        break  # Break inner loop to restart search from beginning
                    else:
                        # Skip this pair and continue searching
                        pos = start + 1
                
                # If no match found in this pass, exit the outer loop
                if not found_match:
                    break
            
            # Strikethrough (~~text~~)
            while '~~' in text:
                start = text.find('~~')
                if start == -1:
                    break
                end = text.find('~~', start + 2)
                if end == -1:
                    break
                before = text[:start]
                content = text[start + 2:end]
                after = text[end + 2:]
                text = before + f'<del>{content}</del>' + after
            
            # Inline code (`text`)
            while '`' in text:
                start = text.find('`')
                if start == -1:
                    break
                end = text.find('`', start + 1)
                if end == -1:
                    break
                before = text[:start]
                content = text[start + 1:end]
                after = text[end + 1:]
                text = before + f'<code>{content}</code>' + after
            
            # Links [text](url)
            while '[' in text and '](' in text and ')' in text:
                start = text.find('[')
                if start == -1:
                    break
                middle = text.find('](', start)
                if middle == -1:
                    break
                end = text.find(')', middle)
                if end == -1:
                    break
                before = text[:start]
                link_text = text[start + 1:middle]
                link_url = text[middle + 2:end]
                after = text[end + 1:]
                text = before + f'<a href="{link_url}">{link_text}</a>' + after
            
            return text
            
        except Exception as e:
            print(f"Error processing inline markdown: {e}")
            return text
    
    def _process_math_expressions(self, text):
        """Process LaTeX mathematical expressions."""
        try:
            # Process block math expressions ($$...$$) first
            # Find all matches and replace from end to beginning to preserve positions
            block_matches = []
            pos = 0
            while True:
                start = text.find('$$', pos)
                if start == -1:
                    break
                end = text.find('$$', start + 2)
                if end == -1:
                    break
                block_matches.append((start, end + 2, text[start + 2:end]))
                pos = end + 2
            
            # Replace from end to beginning
            for start, end, content in reversed(block_matches):
                replacement = f'<div class="math-block">$${content}$$</div>'
                text = text[:start] + replacement + text[end:]
            
            # Process display math expressions (\[...\]) and preserve multiline content
            bracket_matches = []
            pos = 0
            while True:
                start = text.find('\\[', pos)
                if start == -1:
                    break
                end = text.find('\\]', start + 2)
                if end == -1:
                    break
                bracket_matches.append((start, end + 2, text[start + 2:end]))
                pos = end + 2

            for start, end, content in reversed(bracket_matches):
                cleaned = content.replace('<br>', '\n').strip()
                replacement = f'<div class="math-block">$${cleaned}$$</div>'
                text = text[:start] + replacement + text[end:]

            # Process inline math expressions (\(...\))
            inline_matches = []
            pos = 0
            while True:
                start = text.find('\\(', pos)
                if start == -1:
                    break
                end = text.find('\\)', start + 2)
                if end == -1:
                    break
                inline_matches.append((start, end + 2, text[start + 2:end]))
                pos = end + 2
            
            # Replace from end to beginning
            for start, end, content in reversed(inline_matches):
                replacement = f'<span class="math-inline">\\({content}\\)</span>'
                text = text[:start] + replacement + text[end:]
            
            return text
            
        except Exception as e:
            print(f"Error processing math expressions: {e}")
            return text 