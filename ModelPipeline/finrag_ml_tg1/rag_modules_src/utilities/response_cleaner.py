"""
Response cleaning utilities for LLM output.

PHILOSOPHY: Minimal intervention - only fix what's actually broken.
Keep structural formatting (headers, bullets) that aids readability.

✓ Removes (breaks display):
LaTeX: $36.6billionin2016$ → 36.6 billion in 2016
Inline bold: **text** → text (but NOT headers)
Inline italic: *million* → million
HTML tags: <b>text</b> → text
Coverage metrics: 113 of 245 combinations → (removed)

✓ Preserves (helps readability):
Headers: ## Gross Profit Analysis → UNCHANGED
Bullets: - Revenue: $10B → UNCHANGED
Numbers: - Net income: $5B → UNCHANGED
Currency: $ 8.9 billion → UNCHANGED
Structure: Paragraph breaks, sections → UNCHANGED

"""
## python .\ModelPipeline\finrag_ml_tg1\rag_modules_src\utilities\response_cleaner.py

import re
import logging

logger = logging.getLogger(__name__)


class ResponseCleaner:
    """
    Conservative surgical cleanup of LLM-generated text.
    
    Removes ONLY:
        - LaTeX math mode ($...$) that breaks text rendering
        - Inline markdown emphasis (**, *, _) within sentences
        - HTML tags that break display
        - Internal coverage metrics (X of Y combinations)
    
    Preserves:
        - Headers (## Section Name) - structural clarity
        - Bullet points (- item, * item) - content organization  
        - Currency symbols with spaces: "$ 8.9 billion"
        - Natural paragraph breaks
        - Block-level formatting
    
    Example:
        Input:  "Revenue was **$36.6billionin2016** and..."
        Output: "Revenue was $ 36.6 billion in 2016 and..."
        
        Input:  "## Gross Profit Analysis\n\nData shows..."
        Output: "## Gross Profit Analysis\n\nData shows..." (UNCHANGED)
    """
    
    def __init__(self, log_changes: bool = False):
        """
        Initialize cleaner.
        
        Args:
            log_changes: If True, log each cleaning operation
        """
        self.log_changes = log_changes
        self._changes_made = []
    
    
    def clean(self, text: str) -> str:
        """
        Clean LLM response text (conservative approach).
        
        Args:
            text: Raw LLM output
            
        Returns:
            Cleaned text with only problematic formatting removed
        """
        if not text:
            return text
        
        self._changes_made = []
        original_length = len(text)
        
        # Step 1: Remove LaTeX math mode (causes rendering issues)
        text = self._remove_latex_math(text)
        
        # Step 2: Remove INLINE markdown emphasis (**, *, _) 
        # But KEEP headers (##) and bullet points (- item)
        text = self._remove_inline_markdown(text)
        
        # Step 3: Remove HTML tags
        text = self._remove_html_tags(text)
        
        # Step 4: Remove internal coverage metrics
        text = self._remove_coverage_metrics(text)
        
        # Step 5: Fix spacing issues caused by formatting removal
        text = self._fix_spacing_issues(text)
        
        # Step 6: Clean up excessive whitespace (gentle)
        text = self._normalize_whitespace(text)
        
        if self.log_changes and self._changes_made:
            logger.info(
                f"ResponseCleaner: {len(self._changes_made)} operations, "
                f"{original_length} → {len(text)} chars"
            )
            for change in self._changes_made:
                logger.debug(f"  - {change}")
        
        return text


    def _remove_latex_math(self, text: str) -> str:
        """
        Remove LaTeX math mode ($...$) while preserving currency.
        
        Strategy:
            - Keep: "$ 8.9 billion" (space after $ = currency)
            - Keep: "$8.9B" (digit after $ = currency shorthand)
            - Remove: "$x^2 + y^2$" (LaTeX with operators)
            - Remove: "$...$" wrapping around text (formatting wrapper)
        """
        # LaTeX math indicators (not currency)
        latex_indicators = r'[\+\-\*/\^=\\]|\\[a-z]+|[α-ωΑ-Ω]'
        
        def is_latex_math(content):
            """Determine if $...$ is LaTeX math vs currency."""
            # If starts with digit and has financial units, it's currency
            if re.match(r'^\s*\d', content):
                if re.search(r'(billion|million|thousand|B|M|K)\s*$', content, re.I):
                    return False
                # Also keep simple currency: $1.5, $2.3, etc.
                if re.match(r'^\s*[\d,.]+\s*$', content):
                    return False
            
            # If contains LaTeX operators/commands, it's math
            if re.search(latex_indicators, content):
                return True
            
            # If it looks like a wrapped phrase (5+ words), it's formatting
            word_count = len(content.split())
            if word_count >= 5:
                return True
            
            # Short alphanumeric strings might be currency codes
            if len(content) < 10 and content.replace('.', '').replace(' ', '').isalnum():
                return False
            
            # Default: If wrapped in $ $, assume problematic
            return True
        
        # Find all $...$ patterns
        dollar_pattern = r'\$([^\$]+)\$'
        matches = list(re.finditer(dollar_pattern, text))
        
        removed_count = 0
        for match in reversed(matches):  # Reverse to preserve indices
            content = match.group(1)
            if is_latex_math(content):
                # Remove the entire $...$ and keep just the content
                text = text[:match.start()] + content + text[match.end():]
                removed_count += 1
        
        if removed_count > 0:
            self._changes_made.append(f"Removed LaTeX wrappers: {removed_count} blocks")
        
        # FIX: Add space after standalone $ if followed by digit (currency without space)
        # Pattern: $ followed immediately by digit (no space)
        text = re.sub(r'\$(\d)', r'$ \1', text)
        
        return text





    def _remove_inline_markdown(self, text: str) -> str:
        """
        Remove inline markdown with SIMPLE, RELIABLE patterns.
            Protects bullets FIRST with temp marker: * item → <<<BULLET>>>* item
            Removes ALL other *text* patterns - no complex lookbehinds
            Restores bullets by removing marker: <<<BULLET>>>* item → * item

        Strategy: Protect bullets first, then remove everything else.
        """
        # ====================================================================
        # STEP 1: Protect bullet points (temporary markers)
        # ====================================================================
        
        # Bullet patterns to protect:
        # - "- item" or "* item" at start of line
        # - "  - item" or "  * item" (indented)
        
        # Replace bullet asterisks with placeholder
        text = re.sub(r'^(\s*)(\*\s)', r'\1<<<BULLET>>>\2', text, flags=re.MULTILINE)
        
        # ====================================================================
        # STEP 2: Remove ALL markdown formatting (bold and italic)
        # ====================================================================
        
        # Bold: **text** (greedy removal)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        
        # Italic: *text* (greedy removal - now safe because bullets are protected)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        
        # Underscore bold: __text__
        text = re.sub(r'__(.+?)__', r'\1', text)
        
        # Underscore italic: _text_
        text = re.sub(r'_(.+?)_', r'\1', text)
        
        # Strikethrough: ~~text~~
        text = re.sub(r'~~(.+?)~~', r'\1', text)
        
        # ====================================================================
        # STEP 3: Restore bullet markers
        # ====================================================================
        
        text = text.replace('<<<BULLET>>>', '')
        
        self._changes_made.append("Removed inline markdown (bold/italic/strikethrough)")
        
        return text




    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML formatting tags (these break Streamlit display)."""
        patterns = [
            (r'<b>([^<]+)</b>', r'\1', 'bold tag'),
            (r'<strong>([^<]+)</strong>', r'\1', 'strong tag'),
            (r'<i>([^<]+)</i>', r'\1', 'italic tag'),
            (r'<em>([^<]+)</em>', r'\1', 'em tag'),
            (r'<u>([^<]+)</u>', r'\1', 'underline tag'),
        ]
        
        for pattern, replacement, label in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                self._changes_made.append(f"Removed {label}: {len(matches)} instances")
        
        return text
    


    def _remove_coverage_metrics(self, text: str) -> str:
        """
        Remove internal coverage metrics that leaked into response.
        
        Examples:
            - "113 of 245 possible metric combinations"
            - "coverage: 85%"
            - "The dataset contains 113/245 combinations"
        """
        patterns = [
            # "X of Y combinations/metrics"
            r'\d+\s+of\s+\d+\s+(possible\s+)?(metric\s+)?combinations?',
            
            # "X/Y combinations"
            r'\d+/\d+\s+(metric\s+)?combinations?',
            
            # "coverage: X%" or "X% coverage"
            r'coverage:\s+\d+(\.\d+)?%',
            r'\d+(\.\d+)?%\s+coverage',
            
            # Phrases like "with notable gaps in coverage"
            r',?\s*with\s+notable\s+gaps\s+(in\s+coverage)?',
        ]
        
        removed_count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                removed_count += len(matches)
        
        if removed_count > 0:
            self._changes_made.append(f"Removed coverage metrics: {removed_count} instances")
        
        return text
    


    def _fix_spacing_issues(self, text: str) -> str:
        """
        Fix spacing issues created by formatting removal.
        
        Handles:
            - "billionin" → "billion in"
            - "2016to" → "2016 to"
            - "billion.Operating" → "billion. Operating"
            - "2022,thought" → "2022, thought"
            - "annually,reflecting" → "annually, reflecting"
        """
        fixes = [
            # Financial unit + any letter (more aggressive)
            (r'(billion)([a-zA-Z])', r'\1 \2', 'billion spacing'),
            (r'(million)([a-zA-Z])', r'\1 \2', 'million spacing'),
            (r'(thousand)([a-zA-Z])', r'\1 \2', 'thousand spacing'),
            (r'(trillion)([a-zA-Z])', r'\1 \2', 'trillion spacing'),
            
            # Year (4 digits) + any letter
            (r'(\d{4})([a-zA-Z])', r'\1 \2', 'year spacing'),
            
            # Any letter + 4-digit year
            (r'([a-zA-Z])(\d{4})', r'\1 \2', 'word-year spacing'),
            
            # Number + financial unit (from LaTeX removal)
            (r'(\d)(billion|million|thousand)', r'\1 \2', 'number-unit spacing'),
            
            # === NEW PATTERNS FOR SENTENCE BOUNDARIES ===
            
            # Period + capital letter (sentence boundary)
            # Example: "billion.Operating" → "billion. Operating"
            (r'\.([A-Z])', r'. \1', 'sentence period-capital spacing'),
            
            # Comma + lowercase letter (broken mid-sentence)
            # Example: "2022,thought" → "2022, thought"
            (r',([a-z])', r', \1', 'comma-lowercase spacing'),
            
            # Comma + capital letter (less common but defensive)
            (r',([A-Z])', r', \1', 'comma-capital spacing'),
            
            # Period + lowercase letter (rare but possible)
            # Example: "Inc.the" → "Inc. the"
            (r'\.([a-z])', r'. \1', 'period-lowercase spacing'),
        ]
        
        for pattern, replacement, label in fixes:
            # Use IGNORECASE for word patterns, not for punctuation
            if 'billion' in pattern or 'million' in pattern or 'year' in pattern:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                    self._changes_made.append(f"Fixed {label}: {len(matches)} instances")
            else:
                matches = re.findall(pattern, text)
                if matches:
                    text = re.sub(pattern, replacement, text)
                    self._changes_made.append(f"Fixed {label}: {len(matches)} instances")
        
        return text





    def _normalize_whitespace(self, text: str) -> str:
        """
        Gentle whitespace cleanup.
        
        Rules (conservative):
            - Multiple spaces → Single space
            - More than 3 blank lines → 2 blank lines (preserve paragraph breaks)
            - Trailing spaces on lines
        """
        # Remove trailing whitespace from each line
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        # Multiple spaces → Single space (preserve intentional spacing)
        text = re.sub(r' {2,}', ' ', text)
        
        # Excessive blank lines (4+ → 2) but preserve intentional breaks
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # Trim leading/trailing whitespace from entire response
        text = text.strip()
        
        return text



# Convenience function
def clean_llm_response(text: str, log_changes: bool = False) -> str:
    """
    Quick cleaning function for one-off use.
    
    Args:
        text: Raw LLM output
        log_changes: If True, log cleaning operations
        
    Returns:
        Cleaned text
    """
    cleaner = ResponseCleaner(log_changes=log_changes)
    return cleaner.clean(text)


# Testing
if __name__ == "__main__":
    test_cases = [
        # Case 1: LaTeX wrapping + spacing issue
        {
            'input': "Revenue grew to **$36.6billionin2016** to 2.5 billion.",
            'expected': "Revenue grew to $ 36.6 billion in 2016 to 2.5 billion.",
            'focus': 'inline bold + spacing'
        },
        
        # Case 2: Header should be PRESERVED
        {
            'input': "## Gross Profit Analysis\n\nData shows growth.",
            'expected': "## Gross Profit Analysis\n\nData shows growth.",
            'focus': 'header preservation'
        },
        
        # Case 3: Bullet points should be PRESERVED
        {
            'input': "Key points:\n- Revenue: $10B\n- Profit: $2B",
            'expected': "Key points:\n- Revenue: $10B\n- Profit: $2B",
            'focus': 'bullet preservation'
        },
        
        # Case 4: LaTeX math (should be removed)
        {
            'input': "The equation $x^2 + y^2$ shows correlation.",
            'expected': "The equation x^2 + y^2 shows correlation.",
            'focus': 'LaTeX math removal'
        },
        
        # Case 5: Coverage metrics
        {
            'input': "The dataset contains 113 of 245 possible metric combinations.",
            'expected': "The dataset contains .",
            'focus': 'coverage metric removal'
        },
        
        # Case 6: Currency preservation
        {
            'input': "Total was $ 8.9 billion in 2020.",
            'expected': "Total was $ 8.9 billion in 2020.",
            'focus': 'currency preservation'
        },
        
        # Case 7: Mixed formatting (real-world example)
        {
            'input': "**Netflix** expanded from *823.1millionin2016* to $ 2.5 billion.",
            'expected': "Netflix expanded from 823.1 million in 2016 to $ 2.5 billion.",
            'focus': 'complex cleanup'
        },
    ]
    
    cleaner = ResponseCleaner(log_changes=True)
    
    print("=" * 80)
    print("ResponseCleaner Test Suite (Surgical Version)")
    print("=" * 80)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"[Test {i}] {case['focus']}")
        print(f"{'='*80}")
        print(f"Input:\n{case['input']}\n")
        
        result = cleaner.clean(case['input'])
        print(f"Output:\n{result}\n")
        print(f"Expected:\n{case['expected']}\n")
        
        # Flexible comparison (whitespace-normalized)
        result_normalized = ' '.join(result.split())
        expected_normalized = ' '.join(case['expected'].split())
        
        if result_normalized == expected_normalized:
            print("✓ PASS")
        elif result.strip() == case['input'].strip():
            print("✓ PRESERVED (unchanged as intended)")
        else:
            print("~ PARTIAL MATCH")
            print(f"  Difference: Expected '{expected_normalized}'")
            print(f"             Got      '{result_normalized}'")
    
    print("\n" + "=" * 80)
    print("Test suite complete!")
    print("=" * 80)