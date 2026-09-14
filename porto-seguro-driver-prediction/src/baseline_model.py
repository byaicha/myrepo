"""
Porto Seguro's Safe Driver Prediction
Preprocessing-Pipeline + XGBoost-Baseline-Modell (Person 1: Tree-Based Models & Imbalance)
=============================================================================
Autorin: Aicha
"""

import pandas as pd
import numpy as np
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix

import xgboost as xgb

RANDOM_STATE = 42

# -------------------------------------------------------------------
# 1. Daten laden und bereinigen
# -------------------------------------------------------------------
print("1) Lade Datensatz...")
data, meta = arff.loadarff('data/dataset.arff')
df = pd.DataFrame(data)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.decode("utf-8")
    df[col] = df[col].replace("?", "-1")
    df[col] = pd.to_numeric(df[col])

target_col = "target"
print(f"   Shape: {df.shape}")

# -------------------------------------------------------------------
# 2. Feature-Gruppen bestimmen
# -------------------------------------------------------------------
bin_cols = [c for c in df.columns if c.endswith("_bin")]
cat_cols = [c for c in df.columns if c.endswith("_cat")]
remaining_cols = [c for c in df.columns if c not in bin_cols + cat_cols + [target_col]]
calc_cols = [c for c in remaining_cols if "calc" in c]
non_calc_cols = [c for c in remaining_cols if c not in calc_cols]
ordinal_cols = [c for c in non_calc_cols if (df[c].dropna() % 1 == 0).all()]
cont_cols = [c for c in non_calc_cols if c not in ordinal_cols]

# 'calc_'-Features werden verworfen (laut Kaggle-Community kaum Informationsgehalt,
# vgl. Erkenntnis aus der EDA / Logbucheintrag 12.08.2026)
DROP_COLS = calc_cols
print(f"   Verworfene 'calc_'-Features: {len(DROP_COLS)}")

# Spalten mit extrem hoher Fehlquote (>40%, vgl. EDA: ps_car_03_cat, ps_car_05_cat)
# werden ebenfalls verworfen, da Imputation hier kaum sinnvoll ist
high_missing_cols = []
for c in cat_cols:
    missing_share = (df[c] == -1).mean()
    if missing_share > 0.4:
        high_missing_cols.append(c)
print(f"   Verworfene Features wegen hoher Fehlquote (>40%): {high_missing_cols}")

DROP_COLS = DROP_COLS + high_missing_cols
cat_cols = [c for c in cat_cols if c not in high_missing_cols]

feature_cols = [c for c in df.columns if c not in DROP_COLS + [target_col]]
X = df[feature_cols].copy()
y = df[target_col].copy()

# -1 als Platzhalter für fehlende Werte in echte NaN umwandeln,
# damit SimpleImputer sie korrekt erkennt
num_like_cols = [c for c in feature_cols if c not in bin_cols]  # bin-Spalten haben kein -1
for c in num_like_cols:
    X[c] = X[c].replace(-1, np.nan)

print(f"   Finale Feature-Anzahl: {X.shape[1]}")

# -------------------------------------------------------------------
# 3. Train/Test-Split VOR jedem weiteren Schritt
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"2) Train: {X_train.shape}, Test: {X_test.shape}")

# -------------------------------------------------------------------
# 4. Preprocessing-Pipeline (unterschiedliche Behandlung je Feature-Typ)
# -------------------------------------------------------------------
cont_ordinal_cols = [c for c in cont_cols + ordinal_cols if c in feature_cols]
cat_cols_final = [c for c in cat_cols if c in feature_cols]
bin_cols_final = [c for c in bin_cols if c in feature_cols]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    # Kein One-Hot-Encoding nötig: XGBoost kann kategorische Integer-Codes
    # direkt verarbeiten (enable_categorical). Wir belassen sie daher numerisch.
])

binary_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, cont_ordinal_cols),
    ("cat", categorical_transformer, cat_cols_final),
    ("bin", binary_transformer, bin_cols_final),
])

print("3) Preprocessing-Pipeline erstellt "
      f"(numerisch: {len(cont_ordinal_cols)}, kategorisch: {len(cat_cols_final)}, binär: {len(bin_cols_final)})")

# -------------------------------------------------------------------
# 5. Klassenungleichgewicht: scale_pos_weight berechnen
# -------------------------------------------------------------------
neg, pos = np.bincount(y_train.astype(int))
scale_pos_weight = neg / pos
print(f"4) scale_pos_weight = {scale_pos_weight:.2f} (neg={neg}, pos={pos})")

# -------------------------------------------------------------------
# 6. Modell-Pipeline: Preprocessing + XGBoost
# -------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", model),
])

print("5) Trainiere XGBoost-Baseline-Modell...")
pipeline.fit(X_train, y_train)

# -------------------------------------------------------------------
# 7. Evaluation
# -------------------------------------------------------------------
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)

print(f"\n6) Ergebnisse auf dem Test-Set:")
print(f"   AUC-Score: {auc:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------------------------------------
# 8. Feature Importances (Top 15)
# -------------------------------------------------------------------
importances = pipeline.named_steps["model"].feature_importances_
all_feature_names = cont_ordinal_cols + cat_cols_final + bin_cols_final
importance_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop 15 wichtigste Features (laut XGBoost):")
print(importance_df.head(15).to_string(index=False))

print("\nFertig!")