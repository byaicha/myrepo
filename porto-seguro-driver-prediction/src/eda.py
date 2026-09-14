from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Datensatz laden und bereinigen
data, meta = arff.loadarff('data/dataset.arff')
df = pd.DataFrame(data)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.decode('utf-8')
    df[col] = df[col].replace("?", "-1")
    df[col] = pd.to_numeric(df[col])

target_col = "target"
print("Shape:", df.shape)

# 2. Features nach Typ sortieren
bin_cols = [c for c in df.columns if c.endswith("_bin")]
cat_cols = [c for c in df.columns if c.endswith("_cat")]
remaining_cols = [c for c in df.columns if c not in bin_cols + cat_cols + [target_col]]
calc_cols = [c for c in remaining_cols if "calc" in c]
non_calc_cols = [c for c in remaining_cols if c not in calc_cols]
ordinal_cols = [c for c in non_calc_cols if (df[c].dropna() % 1 == 0).all()]
cont_cols = [c for c in non_calc_cols if c not in ordinal_cols]

print("\nBinär:", len(bin_cols))
print("Kategorisch:", len(cat_cols))
print("Ordinal:", len(ordinal_cols))
print("Kontinuierlich:", len(cont_cols))
print("Calc:", len(calc_cols))

# 3. Fehlende Werte (-1) prüfen
missing = (df[cont_cols + ordinal_cols + cat_cols + calc_cols] == -1).sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("\nFehlende Werte pro Spalte:")
print(missing)

plt.figure(figsize=(10, 6))
(missing / len(df) * 100).head(15).plot(kind="barh")
plt.xlabel("Anteil fehlender Werte (%)")
plt.title("Fehlende Werte pro Feature")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("missing_values.png", dpi=150)
plt.close()

# 4. Klassenverteilung
print("\nKlassenverteilung:")
print(df[target_col].value_counts())
print(df[target_col].value_counts(normalize=True) * 100)

plt.figure(figsize=(5, 4))
sns.countplot(x=df[target_col].astype(str))
plt.xlabel("Zielvariable")
plt.title("Klassenverteilung")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.close()

# 5. Korrelationen mit der Zielvariable
corr = df[cont_cols + ordinal_cols + [target_col]].corr()[target_col].drop(target_col)
corr = corr.sort_values(key=abs, ascending=False)
print("\nTop-Korrelationen mit Zielvariable:")
print(corr.head(15))

plt.figure(figsize=(8, 6))
corr.head(15).plot(kind="barh")
plt.xlabel("Korrelation mit Zielvariable")
plt.title("Top Feature-Korrelationen")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("target_correlations.png", dpi=150)
plt.close()

print("\nFertig! Schau in deinem Ordner nach: missing_values.png, class_distribution.png, target_correlations.png")