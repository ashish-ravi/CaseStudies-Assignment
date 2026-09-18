"""Task 2, Part 2: cross-validation, learning curves, fairness."""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference, equalized_odds_difference,
    false_negative_rate, false_positive_rate, selection_rate,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, mean_absolute_error, precision_score, recall_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

import prep

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
OUT = prep.OUTPUT_DIR
OUT.mkdir(parents=True, exist_ok=True)

# Task 1's SVM training cap.
TASK1_SVM_CAP = 15_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_xgb_clf(y_train):
    return XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1)


def make_svm_clf():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                    cache_size=1000, random_state=RANDOM_STATE))])


def make_xgb_reg():
    return XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.08,
                        subsample=0.9, colsample_bytree=0.9,
                        random_state=RANDOM_STATE, n_jobs=-1)


def make_svm_reg():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("reg", SVR(kernel="rbf", C=10.0, gamma="scale", cache_size=1000))])


# --- Cross-validation and sampling ---

def olist_validation(X, y, sens):
    rows = []

    # Task 1 protocol: one stratified 80/20 split.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    xgb = make_xgb_clf(y_tr).fit(X_tr, y_tr)
    single = average_precision_score(y_te, xgb.predict_proba(X_te)[:, 1])
    rows.append(dict(strategy="Single stratified split (Task 1)", model="XGBoost",
                     folds=1, mean=single, sd=np.nan))
    log(f"single split PR-AUC {single:.4f}")

    svm = make_svm_clf()
    X_s, _, y_s, _ = train_test_split(X_tr, y_tr, train_size=TASK1_SVM_CAP,
                                      stratify=y_tr, random_state=RANDOM_STATE)
    svm.fit(X_s, y_s)
    single_svm = average_precision_score(y_te, svm.decision_function(X_te))
    rows.append(dict(strategy="Single stratified split (Task 1)", model="SVM (15k cap)",
                     folds=1, mean=single_svm, sd=np.nan))
    log(f"single split SVM PR-AUC {single_svm:.4f}")

    # Stratified 5-fold, which also gives the spread.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for k, (tr, te) in enumerate(skf.split(X, y)):
        m = make_xgb_clf(y.iloc[tr]).fit(X.iloc[tr], y.iloc[tr])
        scores.append(average_precision_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
        log(f"  stratified fold {k + 1}: {scores[-1]:.4f}")
    rows.append(dict(strategy="Stratified 5-fold CV", model="XGBoost", folds=5,
                     mean=np.mean(scores), sd=np.std(scores, ddof=1)))

    # Grouped by seller: a random split puts the same seller on both sides.
    gkf = GroupKFold(n_splits=5)
    scores_g = []
    for k, (tr, te) in enumerate(gkf.split(X, y, groups=sens["seller_id"])):
        m = make_xgb_clf(y.iloc[tr]).fit(X.iloc[tr], y.iloc[tr])
        scores_g.append(average_precision_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
        log(f"  seller-grouped fold {k + 1}: {scores_g[-1]:.4f}")
    rows.append(dict(strategy="Grouped 5-fold CV (by seller)", model="XGBoost", folds=5,
                     mean=np.mean(scores_g), sd=np.std(scores_g, ddof=1)))

    # Temporal split.
    train_mask = sens["purchase_year"] < 2018
    m = make_xgb_clf(y[train_mask]).fit(X[train_mask], y[train_mask])
    temporal = average_precision_score(y[~train_mask],
                                       m.predict_proba(X[~train_mask])[:, 1])
    rows.append(dict(strategy="Temporal split (train <2018, test 2018)", model="XGBoost",
                     folds=1, mean=temporal, sd=np.nan))
    log(f"temporal PR-AUC {temporal:.4f}")

    pd.DataFrame(rows).to_csv(OUT / "cv_olist.csv", index=False)
    return X_tr, X_te, y_tr, y_te


def food_validation(X, y, sens):
    rows = []
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    xgb = make_xgb_reg().fit(X_tr, y_tr)
    single = mean_absolute_error(y_te, xgb.predict(X_te))
    rows.append(dict(strategy="Single random split (Task 1)", model="XGBoost",
                     folds=1, mean=single, sd=np.nan))
    log(f"food single split MAE {single:.4f}")

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for k, (tr, te) in enumerate(kf.split(X)):
        m = make_xgb_reg().fit(X.iloc[tr], y.iloc[tr])
        scores.append(mean_absolute_error(y.iloc[te], m.predict(X.iloc[te])))
        log(f"  food fold {k + 1}: {scores[-1]:.4f}")
    rows.append(dict(strategy="5-fold CV", model="XGBoost", folds=5,
                     mean=np.mean(scores), sd=np.std(scores, ddof=1)))

    # Same courier appears on both sides of a random split.
    gkf = GroupKFold(n_splits=5)
    scores_g = []
    for k, (tr, te) in enumerate(gkf.split(X, y, groups=sens["courier_id"])):
        m = make_xgb_reg().fit(X.iloc[tr], y.iloc[tr])
        scores_g.append(mean_absolute_error(y.iloc[te], m.predict(X.iloc[te])))
        log(f"  food courier-grouped fold {k + 1}: {scores_g[-1]:.4f}")
    rows.append(dict(strategy="Grouped 5-fold CV (by courier)", model="XGBoost", folds=5,
                     mean=np.mean(scores_g), sd=np.std(scores_g, ddof=1)))

    pd.DataFrame(rows).to_csv(OUT / "cv_food.csv", index=False)
    return X_tr, X_te, y_tr, y_te


# --- Learning curves ---

def olist_learning_curve(X_tr, X_te, y_tr, y_te):
    sizes = [2000, 5000, 10000, 20000, 40000, len(X_tr)]
    rows = []
    for n in sizes:
        if n < len(X_tr):
            Xs, _, ys, _ = train_test_split(X_tr, y_tr, train_size=n, stratify=y_tr,
                                            random_state=RANDOM_STATE)
        else:
            Xs, ys = X_tr, y_tr
        t = time.time()
        m = make_xgb_clf(ys).fit(Xs, ys)
        rows.append(dict(model="XGBoost", n_train=n,
                         score=average_precision_score(y_te, m.predict_proba(X_te)[:, 1]),
                         fit_seconds=time.time() - t))
        log(f"  LC XGBoost n={n}: {rows[-1]['score']:.4f} ({rows[-1]['fit_seconds']:.1f}s)")

        t = time.time()
        s = make_svm_clf().fit(Xs, ys)
        rows.append(dict(model="SVM", n_train=n,
                         score=average_precision_score(y_te, s.decision_function(X_te)),
                         fit_seconds=time.time() - t))
        log(f"  LC SVM     n={n}: {rows[-1]['score']:.4f} ({rows[-1]['fit_seconds']:.1f}s)")
    pd.DataFrame(rows).to_csv(OUT / "learning_curve_olist.csv", index=False)


def food_learning_curve(X_tr, X_te, y_tr, y_te):
    sizes = [2000, 5000, 10000, 15000, 25000, len(X_tr)]
    rows = []
    for n in sizes:
        if n < len(X_tr):
            Xs, _, ys, _ = train_test_split(X_tr, y_tr, train_size=n, random_state=RANDOM_STATE)
        else:
            Xs, ys = X_tr, y_tr
        t = time.time()
        m = make_xgb_reg().fit(Xs, ys)
        rows.append(dict(model="XGBoost", n_train=n,
                         score=mean_absolute_error(y_te, m.predict(X_te)),
                         fit_seconds=time.time() - t))
        log(f"  LC food XGBoost n={n}: {rows[-1]['score']:.4f}")
        t = time.time()
        s = make_svm_reg().fit(Xs, ys)
        rows.append(dict(model="SVM", n_train=n,
                         score=mean_absolute_error(y_te, s.predict(X_te)),
                         fit_seconds=time.time() - t))
        log(f"  LC food SVM     n={n}: {rows[-1]['score']:.4f} ({rows[-1]['fit_seconds']:.1f}s)")
    pd.DataFrame(rows).to_csv(OUT / "learning_curve_food.csv", index=False)


# --- Fairness ---

def olist_fairness(X, y, sens):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    s_te = sens.loc[X_te.index]

    m = make_xgb_clf(y_tr).fit(X_tr, y_tr)
    scores = m.predict_proba(X_te)[:, 1]

    # Task 1's 70% recall operating point, not 0.5.
    threshold = np.quantile(scores[y_te == 1], 0.30)
    y_pred = (scores >= threshold).astype(int)
    log(f"olist fairness threshold {threshold:.4f}, overall recall "
        f"{recall_score(y_te, y_pred):.3f}, precision {precision_score(y_te, y_pred):.3f}")

    metrics = {
        "count": lambda yt, yp: len(yt),
        "base_rate": lambda yt, yp: np.mean(yt),
        "selection_rate": selection_rate,
        "recall": recall_score,
        "precision": precision_score,
        "false_negative_rate": false_negative_rate,
        "false_positive_rate": false_positive_rate,
    }
    for col in ["region", "customer_state"]:
        mf = MetricFrame(metrics=metrics, y_true=y_te, y_pred=y_pred,
                         sensitive_features=s_te[col])
        frame = mf.by_group.reset_index().rename(columns={col: "group"})
        frame.insert(0, "grouping", col)
        frame.to_csv(OUT / f"fairness_olist_{col}.csv", index=False)
        log(f"\n{col}\n{frame.round(3).to_string(index=False)}")

    summary = pd.DataFrame([dict(
        grouping="region",
        demographic_parity_difference=demographic_parity_difference(
            y_te, y_pred, sensitive_features=s_te["region"]),
        equalized_odds_difference=equalized_odds_difference(
            y_te, y_pred, sensitive_features=s_te["region"]),
        overall_recall=recall_score(y_te, y_pred),
        overall_precision=precision_score(y_te, y_pred),
        threshold=threshold)])
    summary.to_csv(OUT / "fairness_olist_summary.csv", index=False)
    log(summary.round(4).to_string(index=False))


def food_fairness(X, y, sens):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    s_te = sens.loc[X_te.index]
    m = make_xgb_reg().fit(X_tr, y_tr)
    pred = m.predict(X_te)

    # Regression: measure error size and direction by group.
    metrics = {
        "count": lambda yt, yp: len(yt),
        "mae": mean_absolute_error,
        "mean_signed_error": lambda yt, yp: np.mean(yp - yt),
        "mean_actual": lambda yt, yp: np.mean(yt),
    }
    frames = []
    for col in ["city_type", "age_band", "rating_band", "vehicle"]:
        mf = MetricFrame(metrics=metrics, y_true=y_te, y_pred=pred,
                         sensitive_features=s_te[col])
        frame = mf.by_group.reset_index().rename(columns={col: "group"})
        frame.insert(0, "grouping", col)
        frames.append(frame)
        log(f"\n{col}\n{frame.round(3).to_string(index=False)}")
    pd.concat(frames).to_csv(OUT / "fairness_food.csv", index=False)


def main():
    log("loading olist")
    X, y, sens = prep.load_olist()
    log(f"olist {X.shape}, late rate {y.mean():.4f}")
    X_tr, X_te, y_tr, y_te = olist_validation(X, y, sens)
    log("olist learning curve")
    olist_learning_curve(X_tr, X_te, y_tr, y_te)
    log("olist fairness")
    olist_fairness(X, y, sens)

    log("loading food")
    Xf, yf, sf = prep.load_food()
    log(f"food {Xf.shape}, mean {yf.mean():.2f}")
    Xf_tr, Xf_te, yf_tr, yf_te = food_validation(Xf, yf, sf)
    log("food learning curve")
    food_learning_curve(Xf_tr, Xf_te, yf_tr, yf_te)
    log("food fairness")
    food_fairness(Xf, yf, sf)
    log("done")


if __name__ == "__main__":
    main()
