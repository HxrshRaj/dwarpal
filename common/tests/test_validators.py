import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.validators import (
    iso6346_check_digit,
    needs_human_review,
    normalize_plate,
    normalize_trailer_id,
    normalize_usdot,
    validate_field,
)


def test_usdot_valid_plain():
    r = normalize_usdot("USDOT 1234567")
    assert r.is_valid_format
    assert r.normalized_text == "1234567"


def test_usdot_fixes_char_confusion():
    # O should become 0 in a digit context
    r = normalize_usdot("USDOT12O4567")
    assert r.normalized_text == "1204567"
    assert r.is_valid_format


def test_usdot_too_short_invalid():
    r = normalize_usdot("USDOT 12")
    assert not r.is_valid_format


def test_usdot_too_long_invalid():
    r = normalize_usdot("USDOT 123456789")
    assert not r.is_valid_format


def test_iso6346_check_digit_known_example():
    # CSQU3054383 is a widely-cited worked example for ISO 6346 (check digit 3)
    assert iso6346_check_digit("CSQU305438") == 3


def test_trailer_id_iso6346_valid():
    r = normalize_trailer_id("CSQU3054383")
    assert r.is_valid_format
    assert "checksum verified" in r.format_note


def test_trailer_id_iso6346_bad_checksum():
    r = normalize_trailer_id("CSQU3054380")
    assert not r.is_valid_format


def test_trailer_id_domestic_fallback():
    r = normalize_trailer_id("ABC123456")
    assert r.is_valid_format
    assert "unverified" in r.format_note


def test_trailer_id_garbage_invalid():
    r = normalize_trailer_id("!!!")
    assert not r.is_valid_format


def test_plate_generic_shape():
    r = normalize_plate("7abc123")
    assert r.is_valid_format
    assert r.normalized_text == "7ABC123"


def test_plate_too_long_invalid():
    r = normalize_plate("ABCDEFGHIJK")
    assert not r.is_valid_format


def test_validate_field_dispatches():
    r = validate_field("usdot", "USDOT 123456")
    assert r.field == "usdot"


def test_validate_field_unknown_class_passthrough():
    r = validate_field("seal", "present")
    assert r.is_valid_format


def test_confidence_gating_below_threshold():
    assert needs_human_review("plate", 0.2) is True


def test_confidence_gating_above_threshold():
    assert needs_human_review("plate", 0.9) is False


def test_confidence_gating_custom_thresholds():
    assert needs_human_review("plate", 0.6, thresholds={"plate": 0.7}) is True
