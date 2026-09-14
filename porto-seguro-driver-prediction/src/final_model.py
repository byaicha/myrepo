"""
Advanced Data Analytics – Porto Seguro's Safe Driver Prediction
Aicha – Tree-Based Models & Imbalance

Reproducible end-to-end analysis:
1) Load/clean ARFF data
2) EDA: class imbalance, missing values, correlations
3) Stratified train/validation/test split
4) XGBoost baseline with class weighting
5) LightGBM and CatBoost comparison
6) SMOTE experiment on a stratified training subset
7) Manual hyperparameter tuning for XGBoost
8) Feature-selection/dimensionality reduction experiment
9) Learning curve
10) Test-set evaluation with ROC-AUC, PR-AUC, F1, recall and precision

The test set is kept untouched until final evaluation.
"""

import argparse, os, time, json
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    recall_score, precision_score, confusion_matrix
)
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42


def load_data(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.decode("utf-8")
        df[col] = df[col].replace("?", "-1")
        df[col] = pd.to_numeric(df[col])
    return df


def prepare_features(df):
    target = "target"
    bin_cols = [c for c in df.columns if c.endswith("_bin")]
    cat_cols = [c for c in df.columns if c.endswith("_cat")]
    remaining = [c for c in df.columns if c not in bin_cols + cat_cols + [target]]
    calc_cols = [c for c in remaining if "calc" in c]
    non_calc = [c for c in remaining if c not in calc_cols]
    ordinal_cols = [c for c in non_calc if (df[c].dropna() % 1 == 0).all()]
    cont_cols = [c for c in non_calc if c not in ordinal_cols]

    high_missing_cols = [
        c for c in cat_cols if (df[c] == -1).mean() > 0.40
    ]
    drop_cols = calc_cols + high_missing_cols
    features = [c for c in df.columns if c not in drop_cols + [target]]

    X = df[features].copy()
    y = df[target].astype(int).copy()
    for c in features:
        if c not in bin_cols:
            X[c] = X[c].replace(-1, np.nan)

    return X, y, bin_cols, cat_cols, ordinal_cols, cont_cols, calc_cols, high_missing_cols


def best_threshold(y_true, proba, start=0.05, stop=0.70, steps=131):
    thresholds = np.linspace(start, stop, steps)
    scores = [f1_score(y_true, proba >= t) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def evaluate(name, model, X_val, y_val, X_test, y_test):
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold = best_threshold(y_val, val_proba)
    test_proba = model.predict_proba(X_test)[:, 1]
    pred = test_proba >= threshold
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, test_proba),
        "pr_auc": average_precision_score(y_test, test_proba),
        "f1": f1_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def main(data_path, out_dir, catboost_sample=200000, smote_sample=100000):
    os.makedirs(out_dir, exist_ok=True)
    df = load_data(data_path)
    X, y, bin_cols, cat_cols, ordinal_cols, cont_cols, calc_cols, high_missing = prepare_features(df)

    # EDA
    class_distribution = y.value_counts().sort_index()
    missing = ((df.drop(columns=["target"]) == -1).mean() * 100).sort_values(ascending=False)
    corr = (
        df[cont_cols + ordinal_cols + ["target"]]
        .corr()["target"].drop("target")
        .sort_values(key=np.abs, ascending=False)
    )
    pd.DataFrame({
        "count": class_distribution,
        "share_percent": class_distribution / len(df) * 100
    }).to_csv(os.path.join(out_dir, "class_distribution.csv"))
    missing.to_csv(os.path.join(out_dir, "missing_values.csv"))
    corr.to_csv(os.path.join(out_dir, "target_correlations.csv"))

    # Split before fitting anything
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.20,
        random_state=RANDOM_STATE, stratify=y_train
    )
    scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

    results = []

    # 1. XGBoost baseline
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc", random_state=RANDOM_STATE,
        n_jobs=-1, tree_method="hist"
    )
    xgb.fit(X_tr, y_tr)
    results.append(evaluate("XGBoost baseline", xgb, X_val, y_val, X_test, y_test))

    # 2. LightGBM
    X_lgb = X.copy()
    cat_final = [c for c in cat_cols if c in X.columns]
    for c in cat_final:
        X_lgb[c] = X_lgb[c].fillna(-1).astype("int8")
    ltr, lval, ltest = X_lgb.loc[X_tr.index], X_lgb.loc[X_val.index], X_lgb.loc[X_test.index]
    lgb = LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
    )
    lgb.fit(ltr, y_tr, categorical_feature=cat_final)
    results.append(evaluate("LightGBM", lgb, lval, y_val, ltest, y_test))

    # 3. CatBoost on a stratified subset to keep runtime reproducible
    X_cat = X.copy()
    for c in cat_final:
        X_cat[c] = X_cat[c].fillna(-1).astype(str)
    cat_indices = [X_cat.columns.get_loc(c) for c in cat_final]
    sample_idx, _ = train_test_split(
        X_tr.index, train_size=min(catboost_sample, len(X_tr)),
        stratify=y_tr, random_state=RANDOM_STATE
    )
    cat = CatBoostClassifier(
        iterations=250, depth=6, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        random_seed=RANDOM_STATE, verbose=False, thread_count=-1,
        class_weights=[1.0, scale_pos_weight],
        allow_writing_files=False
    )
    cat.fit(X_cat.loc[sample_idx], y.loc[sample_idx], cat_features=cat_indices)
    results.append(evaluate(
        "CatBoost (stratified subset)", cat,
        X_cat.loc[X_val.index], y_val, X_cat.loc[X_test.index], y_test
    ))

    # 4. SMOTE experiment on a subset
    sm_idx, _ = train_test_split(
        X_tr.index, train_size=min(smote_sample, len(X_tr)),
        stratify=y_tr, random_state=RANDOM_STATE
    )
    imputer = SimpleImputer(strategy="median")
    X_sm = imputer.fit_transform(X.loc[sm_idx])
    y_sm = y.loc[sm_idx]
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_sm, y_sm)
    sm_xgb = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", random_state=RANDOM_STATE,
        n_jobs=-1, tree_method="hist"
    )
    sm_xgb.fit(X_res, y_res)
    results.append(evaluate(
        "XGBoost + SMOTE (subset)", sm_xgb,
        imputer.transform(X_val), y_val, imputer.transform(X_test), y_test
    ))

    # 5. Manual XGBoost hyperparameter tuning
    configs = [
        dict(n_estimators=500, max_depth=3, learning_rate=0.03,
             subsample=0.9, colsample_bytree=0.9, min_child_weight=5),
        dict(n_estimators=300, max_depth=5, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.8, min_child_weight=3),
        dict(n_estimators=400, max_depth=4, learning_rate=0.03,
             subsample=0.9, colsample_bytree=0.8, min_child_weight=5),
    ]
    candidates = []
    for cfg in configs:
        m = XGBClassifier(
            **cfg, scale_pos_weight=scale_pos_weight,
            eval_metric="auc", random_state=RANDOM_STATE,
            n_jobs=-1, tree_method="hist"
        )
        m.fit(X_tr, y_tr)
        pv = m.predict_proba(X_val)[:, 1]
        candidates.append((roc_auc_score(y_val, pv), m))
    best_model = max(candidates, key=lambda z: z[0])[1]
    results.append(evaluate("XGBoost tuned", best_model, X_val, y_val, X_test, y_test))

    # 6. Feature selection as dimensionality-reduction experiment
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)
    feature_importance.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
    top20 = feature_importance.head(20)["feature"].tolist()
    reduced = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc", random_state=RANDOM_STATE,
        n_jobs=-1, tree_method="hist"
    )
    reduced.fit(X_tr[top20], y_tr)
    results.append(evaluate("XGBoost top-20 features", reduced,
                            X_val[top20], y_val, X_test[top20], y_test))

    pd.DataFrame(results).drop(columns=["confusion_matrix"]).to_csv(
        os.path.join(out_dir, "model_comparison.csv"), index=False
    )
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Datensatz:", df.shape)
    print("Features nach Reduktion:", X.shape[1])
    print("Klassen:", class_distribution.to_dict())
    print("scale_pos_weight:", round(scale_pos_weight, 3))
    print("\nModellvergleich:")
    print(pd.DataFrame(results).drop(columns=["confusion_matrix"]).to_string(index=False))
    print("\nTop-20 Features:")
    print(top20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset.arff")
    parser.add_argument("--out", default="results")
    parser.add_argument("--catboost-sample", type=int, default=200000)
    parser.add_argument("--smote-sample", type=int, default=100000)
    args = parser.parse_args()
    main(args.data, args.out, args.catboost_sample, args.smote_sample)
