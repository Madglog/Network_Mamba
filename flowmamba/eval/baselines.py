"""Classical baselines on aggregated flow statistics. No PyTorch dependency.

Each baseline answers a specific question from the proposal:

  * XGBoost           -- the classical *floor* for supervised classification.
    The deep detector earns its complexity only if it beats gradient-boosted
    trees on what XGBoost cannot target (zero-day, cross-dataset, adversarial).
  * Isolation Forest  -- the classical unsupervised floor for the anomaly head.

These run on the conventional ~80-statistic aggregated vectors produced by
:func:`flowmamba.data.features.aggregate_flows`, which is also the representation
control isolating the gain from keeping packet order.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def xgboost_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    n_estimators: int = 300,
    max_depth: int = 6,
    seed: int = 1337,
) -> Dict:
    """Train an XGBoost classifier on aggregated stats; return preds + metrics.

    Real captures contain only a *subset* of the 8 categories, so the labels are
    typically non-contiguous (e.g. ``{0, 1, 3, 7}``). XGBoost requires contiguous
    ``0..K-1`` classes, so the present classes are remapped to a dense space
    internally; ``class_names`` and the probability columns are subset to match,
    and predictions are mapped back to the original label space on return. Test
    samples whose class never appears in training cannot be scored by a supervised
    classifier and are excluded from the supervised metric (they belong to the
    anomaly / zero-day evaluation instead).
    """
    from xgboost import XGBClassifier

    from flowmamba.eval.metrics import classification_metrics, pr_auc_per_class

    classes = np.unique(y_train)                       # classes XGBoost will learn
    remap = {int(c): i for i, c in enumerate(classes)}
    y_tr = np.array([remap[int(v)] for v in y_train], dtype=np.int64)

    seen = np.isin(y_test, classes)
    x_te = x_test[seen]
    y_te = np.array([remap[int(v)] for v in y_test[seen]], dtype=np.int64)
    sub_names = [class_names[int(c)] for c in classes] if class_names is not None else None

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(classes),
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_train, y_tr)
    proba = clf.predict_proba(x_te)                    # columns aligned with `classes`
    pred_dense = proba.argmax(axis=1)
    pred = classes[pred_dense]                          # back to original label space

    return {
        "model": clf,
        "pred": pred,
        "proba": proba,
        "classes": classes,
        "n_test_dropped": int((~seen).sum()),
        "metrics": classification_metrics(y_te, pred_dense, sub_names),
        "pr_auc": pr_auc_per_class(y_te, proba, sub_names),
    }


def isolation_forest_baseline(
    x_train_benign: np.ndarray,
    x_test: np.ndarray,
    y_test_is_attack: np.ndarray,
    contamination: float = 0.01,
    n_estimators: int = 200,
    seed: int = 1337,
) -> Dict:
    """Fit Isolation Forest on benign aggregated stats; score the test set.

    Trained one-class on benign only, matching how the deep anomaly head is fit.
    Higher returned score == more anomalous (sign-flipped from sklearn's
    ``score_samples`` so it aligns with the deep SVDD convention).
    """
    from sklearn.ensemble import IsolationForest

    from flowmamba.eval.metrics import anomaly_detection_metrics, binary_rates

    iforest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    iforest.fit(x_train_benign)
    anomaly_score = -iforest.score_samples(x_test)   # higher == more anomalous
    flagged = iforest.predict(x_test) == -1          # sklearn: -1 == outlier

    return {
        "model": iforest,
        "anomaly_score": anomaly_score,
        "flagged": flagged,
        "metrics": anomaly_detection_metrics(y_test_is_attack, anomaly_score),
        "rates": binary_rates(y_test_is_attack, flagged),
    }
