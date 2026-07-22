from pathlib import Path

import numpy as np

from scripts.calibrated_fusion_detector import compute_confusion_metrics, split_paths_three_way


def test_split_paths_three_way_is_disjoint_and_complete():
    paths = [Path(f"img_{i}.png") for i in range(12)]

    train, val, test = split_paths_three_way(paths, seed=7)

    assert len(train) + len(val) + len(test) == len(paths)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0


def test_confusion_metrics_compute_expected_values():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])

    metrics = compute_confusion_metrics(y_true, y_pred)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["tpr"] == 0.5
    assert metrics["fpr"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["f1"] == 0.5
