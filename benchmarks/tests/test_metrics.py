import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.metrics import (
    average_precision_at_iou,
    character_error_rate,
    exact_match,
    iou_xywh,
    levenshtein,
    mean_average_precision,
)


def test_iou_identical_boxes_is_one():
    box = [10, 10, 20, 20]
    assert iou_xywh(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou_xywh([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0


def test_iou_partial_overlap():
    # two 10x10 boxes overlapping by 5x10 -> intersection 50, union 150
    iou = iou_xywh([0, 0, 10, 10], [5, 0, 10, 10])
    assert abs(iou - 50 / 150) < 1e-6


def test_ap_perfect_prediction_is_one():
    gt = [[0, 0, 10, 10]]
    preds = [([0, 0, 10, 10], 0.9)]
    assert average_precision_at_iou(preds, gt) == 1.0


def test_ap_no_predictions_no_gt_is_one():
    assert average_precision_at_iou([], []) == 1.0


def test_ap_false_positive_only_is_zero():
    gt = []
    preds = [([0, 0, 10, 10], 0.9)]
    assert average_precision_at_iou(preds, gt) == 0.0


def test_ap_missed_detection_is_zero():
    gt = [[0, 0, 10, 10]]
    preds = []
    assert average_precision_at_iou(preds, gt) == 0.0


def test_mean_ap_averages_correctly():
    assert mean_average_precision([1.0, 0.0, 0.5]) == 0.5


def test_mean_ap_empty_is_zero():
    assert mean_average_precision([]) == 0.0


def test_exact_match_case_insensitive():
    assert exact_match("abc123", "ABC123")


def test_exact_match_whitespace_stripped():
    assert exact_match(" ABC123 ", "ABC123")


def test_exact_match_false_on_difference():
    assert not exact_match("ABC124", "ABC123")


def test_exact_match_ignores_punctuation_like_the_production_validator():
    # e.g. EasyOCR emitting "YG9-X2G" for ground truth "YG9X2G" should count
    # as a match, because common/validators.py strips the hyphen before an
    # operator ever sees it -- exact_match should reflect deployed behavior.
    assert exact_match("YG9-X2G", "YG9X2G")
    assert exact_match("YGg suntrup K26 3", "YGGSUNTRUPK263")


def test_levenshtein_identical_is_zero():
    assert levenshtein("ABC", "ABC") == 0


def test_levenshtein_one_substitution():
    assert levenshtein("ABC", "ABD") == 1


def test_levenshtein_empty_strings():
    assert levenshtein("", "ABC") == 3
    assert levenshtein("ABC", "") == 3


def test_cer_perfect_match_is_zero():
    assert character_error_rate("ABC123", "abc123") == 0.0


def test_cer_one_char_off():
    cer = character_error_rate("ABC124", "ABC123")
    assert abs(cer - 1 / 6) < 1e-6


def test_cer_empty_gt_and_pred_is_zero():
    assert character_error_rate("", "") == 0.0


def test_cer_empty_gt_nonempty_pred_is_one():
    assert character_error_rate("ABC", "") == 1.0
