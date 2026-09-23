"""La prova indipendente: URL che il modello non ha mai visto e che vengono da
fonti diverse da quelle di addestramento.
  openphish   phishing verificato, raccolto oggi        -> quanti ne prende
  hackernews  link postati oggi da persone vere         -> quanti buoni blocca
  urlhaus     URL di malware (minaccia diversa)         -> generalizza o no
"""
import json, pathlib, sys
import numpy as np, pandas as pd, joblib
import features as F

ROOT = pathlib.Path(__file__).parent
SUFF = "_naturale" if "--naturale" in sys.argv else ""
OUT = ROOT / f"outputs{SUFF}"
model = joblib.load(OUT / "model.joblib")
th = json.load(open(OUT / "thresholds.json"))
h = pd.read_csv(ROOT / "data/holdout_fresh.csv")
X = F.featurize(h.url.tolist())
h["p"] = model.predict_proba(X)[:, 1]

rows = []
for src, g in h.groupby("source"):
    r = {"fonte": src, "n": len(g), "label": int(g.label.iloc[0]), "score_mediano": round(g.p.median(), 3)}
    for tag, t in (("0.50", .5), ("FPR1%", th["threshold_fpr1"]), ("FPR0.1%", th["threshold_fpr01"])):
        flag = (g.p >= t).mean()
        r[f"segnalati@{tag}"] = f"{flag:.1%}"
    rows.append(r)
rep = pd.DataFrame(rows)
print(rep.to_string(index=False))
print("\nnota: per le fonti con label 1 'segnalati' = recall; per hackernews = falsi positivi")

bad = h[(h.source == "hackernews") & (h.p >= th["threshold_fpr1"])].nlargest(12, "p")
print(f"\n--- link buoni che il modello sbaglia (i primi {len(bad)}):")
for _, r in bad.iterrows(): print(f"  {r.p:.3f}  {r.url[:110]}")
miss = h[(h.source == "openphish") & (h.p < th["threshold_fpr1"])].nsmallest(12, "p")
print(f"\n--- phishing che si e' fatto sfuggire ({(h[h.source=='openphish'].p < th['threshold_fpr1']).sum()} su {(h.source=='openphish').sum()}):")
for _, r in miss.iterrows(): print(f"  {r.p:.3f}  {r.url[:110]}")
h.to_csv(OUT / "holdout_scored.csv", index=False)
