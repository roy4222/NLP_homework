from eval.metrics import compute_multilabel_metrics, compute_per_label_f1


def test_compute_multilabel_metrics_for_perfect_predictions():
    metrics = compute_multilabel_metrics([["a"], ["a", "b"]], [["a"], ["a", "b"]])

    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["hamming_loss"] == 0.0


def test_compute_multilabel_metrics_includes_predicted_only_labels_without_warning(recwarn):
    metrics = compute_multilabel_metrics([["a"]], [["a", "b"]])

    assert not recwarn
    assert metrics["precision"] == 0.5


def test_compute_per_label_f1_reports_support_and_scores():
    rows = compute_per_label_f1([["a"], ["a", "b"]], [["a"], ["a"]])

    assert rows == [
        {"label": "a", "f1": 1.0, "support": 2},
        {"label": "b", "f1": 0.0, "support": 1},
    ]
