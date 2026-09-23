"""Ordina una lista di contatti dal piu' probabile al meno probabile.
  python score.py lista.csv            -> stampa i primi 20 + salva outputs/scored.csv
Il CSV deve avere le stesse colonne del dataset (senza 'y' e senza 'duration').
Senza argomenti usa il test set come demo.
"""
import sys, pathlib, joblib, pandas as pd

ROOT = pathlib.Path(__file__).parent
model = joblib.load(ROOT / "outputs/model.joblib")
THRESHOLD = 0.2373  # soglia che massimizza F1 sul test set, vedi outputs/metrics.json

src = sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/bank-additional-full.csv"
df = pd.read_csv(src, sep=";").drop(columns=["y", "duration"], errors="ignore")

df["probabilita"] = model.predict_proba(df)[:, 1]
df["chiamare"] = df["probabilita"] >= THRESHOLD
out = df.sort_values("probabilita", ascending=False)
out.to_csv(ROOT / "outputs/scored.csv", index=False)

print(f"{len(out)} contatti, {out['chiamare'].sum()} sopra soglia {THRESHOLD}")
print(out[["age", "job", "contact", "month", "poutcome", "probabilita"]].head(20).to_string(index=False))
print("\n-> outputs/scored.csv")
