"""Field-specific validation and normalization for OCR output.

Format sources (verified before writing any regex — see docs/data.md
"Format references" for full citations):

  USDOT number
    49 CFR 390.21(b)(2) (govinfo.gov CFR-2024-title49-vol5-sec390-21):
    "the identification number issued by FMCSA ... preceded by the letters
    USDOT." The regulation itself does NOT specify a digit count. FMCSA's
    public SAFER registration system observably assigns numbers in the
    6-8 digit range as of 2026 (informal observation, not a cited spec —
    flagged as such). We validate 5-8 digits to be permissive and flag
    outside that range as low-confidence rather than hard-rejecting.

  Trailer ID
    No single US domestic standard exists. Intermodal chassis/containers
    follow ISO 6346 (4-letter prefix: 3-letter owner code + 1 equipment
    category letter, + 6-digit serial + 1 mod-11 check digit). Domestic
    dry-van trailers commonly use carrier-specific alphanumeric IDs with
    no public standard. We validate ISO 6346 strictly (checksum) when the
    shape matches, and fall back to a permissive alphanumeric check
    otherwise, explicitly marked "unverified format" — this is an honest
    reporting choice, not a hidden guess.

  License plate
    No single national format exists; plates are issued by individual US
    states/provinces with different lengths and character rules. We only
    validate a generic alphanumeric shape (2-8 chars) and do not claim to
    validate any specific state's format. See docs/data.md.

  Seal presence
    Not a text field — boolean detection confidence only. No format
    validation applies.
"""
import re
from dataclasses import dataclass
from typing import Optional

_CHAR_CONFUSION_TO_DIGIT = {"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8", "Z": "2"}
_DIGIT_TO_CHAR_CONFUSION = {"0": "O", "1": "I", "5": "S", "8": "B"}

ISO6346_LETTER_VALUES = {}
_v = 10
for _c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    while _v % 11 == 0:
        _v += 1
    ISO6346_LETTER_VALUES[_c] = _v
    _v += 1


@dataclass
class ValidationResult:
    field: str
    raw_text: str
    normalized_text: str
    is_valid_format: bool
    format_note: str


def _fix_digit_confusions(text: str) -> str:
    return "".join(_CHAR_CONFUSION_TO_DIGIT.get(c, c) for c in text)


def normalize_usdot(raw: str) -> ValidationResult:
    text = raw.strip().upper()
    text = re.sub(r"^USDOT[:\s]*", "", text)
    digits = _fix_digit_confusions(text)
    digits = re.sub(r"[^0-9]", "", digits)
    valid = bool(re.fullmatch(r"\d{5,8}", digits))
    note = "ok" if valid else f"expected 5-8 digits after USDOT prefix, got '{digits}' ({len(digits)} chars)"
    return ValidationResult("usdot", raw, digits, valid, note)


def iso6346_check_digit(owner_and_serial: str) -> Optional[int]:
    if len(owner_and_serial) != 10 or not owner_and_serial[:4].isalpha() or not owner_and_serial[4:].isdigit():
        return None
    total = 0
    for i, ch in enumerate(owner_and_serial):
        val = ISO6346_LETTER_VALUES.get(ch) if ch.isalpha() else int(ch)
        if val is None:
            return None
        total += val * (2 ** i)
    return total % 11 % 10


def normalize_trailer_id(raw: str) -> ValidationResult:
    text = re.sub(r"[^A-Za-z0-9]", "", raw.strip().upper())
    # try ISO 6346 (4 letters + 6 digits + 1 check digit = 11 chars)
    if len(text) == 11 and text[:4].isalpha() and text[4:].isdigit():
        expected = iso6346_check_digit(text[:10])
        actual = int(text[10])
        if expected is not None and expected == actual:
            return ValidationResult("trailer_id", raw, text, True, "ok (ISO 6346 checksum verified)")
        return ValidationResult(
            "trailer_id", raw, text, False, f"ISO 6346 shape but checksum mismatch (expected {expected}, got {actual})"
        )
    # fall back: permissive domestic alphanumeric, unverified
    if re.fullmatch(r"[A-Z]{2,4}\d{4,7}", text):
        return ValidationResult("trailer_id", raw, text, True, "ok (domestic format, unverified — no public standard exists)")
    return ValidationResult("trailer_id", raw, text, False, f"does not match ISO 6346 or common domestic shapes: '{text}'")


def normalize_plate(raw: str) -> ValidationResult:
    text = re.sub(r"[^A-Za-z0-9]", "", raw.strip().upper())
    valid = bool(re.fullmatch(r"[A-Z0-9]{2,8}", text))
    note = "ok (generic shape only — no single national plate format exists)" if valid else f"outside 2-8 alphanumeric chars: '{text}'"
    return ValidationResult("plate", raw, text, valid, note)


VALIDATORS = {
    "usdot": normalize_usdot,
    "trailer_id": normalize_trailer_id,
    "plate": normalize_plate,
}


def validate_field(field_class: str, raw_text: str) -> ValidationResult:
    fn = VALIDATORS.get(field_class)
    if fn is None:
        return ValidationResult(field_class, raw_text, raw_text.strip(), True, "no validator defined for this field")
    return fn(raw_text)


# --- Confidence gating -------------------------------------------------
# Default thresholds below are PLACEHOLDERS until benchmarks/tune_thresholds.py
# computes a real precision/coverage curve from OCR-on-real-data results and
# overwrites confidence_thresholds.json. Do not treat these as measured.
DEFAULT_CONFIDENCE_THRESHOLDS = {
    "plate": 0.5,
    "usdot": 0.5,
    "trailer_id": 0.5,
    "seal": 0.5,
}


def needs_human_review(field_class: str, confidence: float, thresholds: dict = None) -> bool:
    thresholds = thresholds or DEFAULT_CONFIDENCE_THRESHOLDS
    threshold = thresholds.get(field_class, 0.5)
    return confidence < threshold
