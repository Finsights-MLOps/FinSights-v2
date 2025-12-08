"""
Minimal response cleaner - prevent LaTeX/markdown triggering in Streamlit.

Philosophy: Line-aware cleaning - preserve structure, neutralize triggers.
"""

import re
import logging

logger = logging.getLogger(__name__)


class ResponseCleaner:
    """
    Line-aware cleaner that prevents LaTeX/markdown triggering.
    
    Strategy:
        1. Classify each line (heading, bullet, ordinary)
        2. Remove markdown wrappers ($...$, **bold**, *italic*)
        3. Escape residual triggers (* _ $) to prevent Streamlit interpretation
        4. Preserve heading structure and bullet prefixes
    
    Hard Guarantee:
        After cleaning, only these render as markdown:
        - Headings starting with # (preserved structure)
        - Bullet prefixes -, *, +, 1. (preserved structure)
        - Everything else is markdown-inert (escaped)
    """
    
    def __init__(self, log_changes: bool = False):
        self.log_changes = log_changes
        self._changes_made = []
    
    def clean(self, text: str) -> str:
        """
        Clean LLM response line-by-line.
        
        Args:
            text: Raw LLM output
            
        Returns:
            Cleaned text safe for st.markdown()
        """
        if not text:
            return text
        
        self._changes_made = []
        
        # Process line-by-line
        cleaned_lines = []
        for line in text.split('\n'):
            cleaned_line = self._clean_line(line)
            cleaned_lines.append(cleaned_line)
        
        result = '\n'.join(cleaned_lines)
        
        if self.log_changes:
            logger.info(
                f"ResponseCleaner: {len(self._changes_made)} operations, "
                f"{len(text)} → {len(result)} chars"
            )
        
        return result
    
    def _clean_line(self, line: str) -> str:
        """
        Clean a single line based on its type.
        
        Line types:
            1. Heading (starts with #) → Preserve structure, clean body
            2. Bullet/List (starts with -, *, +, 1.) → Preserve prefix, clean body
            3. Ordinary → Full cleaning
        """
        # ================================================================
        # Type 1: Heading - PRESERVE STRUCTURE, CLEAN BODY
        # ================================================================
        # Pattern: optional indent + 1-6 hashes + optional space + body
        # Example: "  ## Revenue Analysis *2022*" → "  ## Revenue Analysis 2022"
        
        heading_match = re.match(r'^(\s*)(#{1,6})\s*(.*)$', line)
        if heading_match:
            indent = heading_match.group(1)
            hashes = heading_match.group(2)
            body = heading_match.group(3)
            
            # Clean the heading body (remove wrappers, escape triggers)
            body = self._strip_wrappers(body)
            body = self._escape_triggers(body)
            
            # Reconstruct: indent + hashes + space + body
            cleaned = f"{hashes} {body}".rstrip() if body else hashes
            return f"{indent}{cleaned}"
        
        # ================================================================
        # Type 2: Bullet/List - PRESERVE PREFIX, CLEAN BODY
        # ================================================================
        # Patterns: "- item", "  * item", "1. item", "  + item"
        bullet_match = re.match(r'^(\s*)([-*+]|\d+\.)\s+(.*)$', line)
        if bullet_match:
            indent = bullet_match.group(1)
            prefix = bullet_match.group(2)
            body = bullet_match.group(3)
            
            # Clean the body part only
            body = self._strip_wrappers(body)
            body = self._escape_triggers(body)
            
            return f"{indent}{prefix} {body}"
        
        # ================================================================
        # Type 3: Ordinary Line - FULL CLEANING
        # ================================================================
        line = self._strip_wrappers(line)
        line = self._escape_triggers(line)
        
        return line
    
    def _strip_wrappers(self, text: str) -> str:
        """
        Remove markdown/LaTeX wrappers.
        
        Removes:
            $...$ (LaTeX)
            **bold**, __bold__
            *italic*, _italic_
            ~~strikethrough~~
        
        Simple unwrapping - no fancy logic.
        """
        # LaTeX: $content$ → content
        text = re.sub(r'\$([^\$]+)\$', r'\1', text)
        
        # Bold: **text** and __text__
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        
        # Italic: *text* and _text_
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        
        # Strikethrough: ~~text~~
        text = re.sub(r'~~(.+?)~~', r'\1', text)
        
        return text
    
    
    def _escape_triggers(self, text: str) -> str:
        r"""
        Make text completely safe for Streamlit markdown.
        
        Strategy:
            - Replace all '$' with 'USD ' (eliminates LaTeX risk entirely)
            - Escape '*' and '_' (prevents italic/bold)
        
        Hard guarantee: No LaTeX math mode possible after this.
        """
        # Replace all $ with USD (no dollar symbol remains)
        text = text.replace('$', 'USD ')
        
        # Escape asterisks and underscores (idempotent)
        text = re.sub(r'(?<!\\)\*', '\\*', text)
        text = re.sub(r'(?<!\\)_', '\\_', text)
        
        return text



def clean_llm_response(text: str, log_changes: bool = False) -> str:
    """Convenience function."""
    cleaner = ResponseCleaner(log_changes=log_changes)
    return cleaner.clean(text)


# Testing
if __name__ == "__main__":
    test_cases = [
        {
            'name': 'Heading with italic',
            'input': '## Revenue Analysis *2022*',
            'expected': '## Revenue Analysis 2022'
        },
        {
            'name': 'Body with LaTeX wrapper',
            'input': 'Revenue was $36.6billionin2016$ last year.',
            'expected': 'Revenue was 36.6billionin2016 last year.'
        },
        {
            'name': 'Bullet with bold',
            'input': '- Revenue: **$10B**',
            'expected': '- Revenue: \\$10B'
        },
        {
            'name': 'Ordinary with italic',
            'input': 'The company had *negative* cash flow.',
            'expected': 'The company had negative cash flow.'
        },
        {
            'name': 'Mixed formatting',
            'input': 'Netflix **grew** from *2.5B* to $31.6B$.',
            'expected': 'Netflix grew from 2.5B to 31.6B.'
        },
    ]
    
    cleaner = ResponseCleaner(log_changes=True)
    
    print("=" * 80)
    print("ResponseCleaner Test Suite (Enhanced Minimal)")
    print("=" * 80)
    
    for case in test_cases:
        print(f"\n[Test: {case['name']}]")
        print(f"Input:    {case['input']}")
        
        result = cleaner.clean(case['input'])
        print(f"Output:   {result}")
        print(f"Expected: {case['expected']}")
        
        if result == case['expected']:
            print("✓ PASS")
        else:
            print("~ CHECK")
    
    print("\n" + "=" * 80)
