"""
Shared token-estimation helpers.

``estimate_tokens_from_bytes`` is a coarse ~4 bytes/token ratio for payloads
that are only available as a byte count (telemetry).

``estimate_tokens_from_text`` approximates how a byte-level BPE tokenizer
splits the text, and is what the canvas-state budget uses. The ratio above
badly underestimates number-heavy text: Qwen-family models (the default local
model) spend one token per digit, so ``140.8556708123603`` is 17 tokens, not 4.
The heuristic follows the Qwen/GPT-4 pre-tokenizer:

1. every digit is one token;
2. a run of letters (with one leading space or symbol) is one token per 6 letters;
3. a run of punctuation is one token per 3 characters;
4. whitespace and newline runs are one token;
5. CJK and other wide characters (U+2E80 and up) are one token each, since
   BPE vocabularies hold few multi-character CJK tokens.

Checked against the real Qwen tokenizer on captured canvas states (raw JSON,
compact JSON and the text format) it lands within -5%..+11%, and it
overestimates plain prose, which keeps budgets on the safe side. OpenAI
tokenizers group up to three digits per token, so for cloud models the
estimate is an upper bound.
"""

from __future__ import annotations

import math
import re

_ESTIMATED_TOKEN_RATIO = 4
_LETTERS_PER_TOKEN = 6
_SYMBOLS_PER_TOKEN = 3
# Characters from here on (CJK radicals, kana, hangul, ideographs, ...) count one token each.
_WIDE_CHARACTER_START = 0x2E80

# Simplified pre-tokenizer: contractions, words, single digits, symbol runs, whitespace.
_PRE_TOKEN = re.compile(
    r"(?P<contraction>'(?:[sStTmMdD]|[rR][eE]|[vV][eE]|[lL][lL]))"
    r"|(?P<word>(?:[^\r\n\w]|_)?[^\W\d_]+)"
    r"|(?P<digit>\d)"
    r"|(?P<symbols> ?(?:[^\s\w]|_)+)[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def estimate_tokens_from_bytes(payload_bytes: int) -> int:
    if payload_bytes <= 0:
        return 0
    return max(1, payload_bytes // _ESTIMATED_TOKEN_RATIO)


def estimate_tokens_from_text(text: str) -> int:
    """Estimate the token count of ``text`` (digits count one token each)."""
    if not text:
        return 0
    total = 0
    for match in _PRE_TOKEN.finditer(text):
        word = match.group("word")
        symbols = match.group("symbols")
        if word is not None:
            total += _run_tokens(word.strip(), _LETTERS_PER_TOKEN)
        elif symbols is not None:
            total += _run_tokens(symbols.strip(), _SYMBOLS_PER_TOKEN)
        else:
            total += 1
    return total


def _run_tokens(run: str, characters_per_token: int) -> int:
    """Tokens for a letter or symbol run: wide characters count one each, the rest are grouped."""
    wide = sum(1 for character in run if ord(character) >= _WIDE_CHARACTER_START)
    narrow = len(run) - wide
    return max(1, wide + math.ceil(narrow / characters_per_token))
