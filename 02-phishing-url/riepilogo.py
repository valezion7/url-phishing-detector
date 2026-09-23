"""Raccoglie in un unico posto tutti i numeri finali, cosi' README e pagina pubblica
non li riportano a mano."""
import json, pathlib, sys
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import features as F

ROOT = pathlib.Path(__file__).parent
SUFF = "_naturale" if "--naturale" in sys.argv else ""
OUT = ROOT / f"outputs{SUFF}"
BENIGNI = ("hackernews", "lobsters", "hackernews_recenti", "hackernews_storico")

m = json.load(open(OUT / "metrics_splits.json"))
tab = pd.DataFrame({nm: {"AUC casuale": m["casuale"][nm]["roc_auc"],
                         "AUC per dominio": m["per_dominio"][nm]["roc_auc"],
                         "PR-AUC per dominio": m["per_dominio"][nm]["pr_auc"],
                         "accuratezza": m["per_dominio"][nm]["accuracy@0.5"],
                         "phishing preso @FPR1%": m["per_dominio"][nm]["recall@fpr1%"],
                         "phishing preso @FPR0.1%": m["per_dominio"][nm]["recall@fpr0.1%"],
                         "secondi": m["per_dominio"][nm]["fit_s"]} for nm in m["per_dominio"]}).T
print("=== test interno, split per dominio ===")
print(tab.round(4).to_string())

h = pd.read_csv(ROOT / "data/holdout_fresh.csv")
X = F.featurize(h.url.tolist())
p = joblib.load(OUT / "model.joblib").predict_proba(X)[:, 1]
h["p"] = p
buoni, phish = h.source.isin(BENIGNI), h.source == "openphish"
fresco = buoni | phish
fpr, tpr, th = roc_curve(h.label[fresco], p[fresco])
righe = []
for f in (.005, .01, .02, .05, .10):
    i = np.searchsorted(fpr, f, "right") - 1
    righe.append({"buoni bloccati": f"{fpr[i]:.2%}", "phishing preso": f"{tpr[i]:.1%}",
                  "soglia": round(float(th[i]), 3),
                  "malware URLhaus preso": f"{(p[h.source=='urlhaus'] >= th[i]).mean():.1%}"})
print(f"\n=== holdout indipendente ({buoni.sum()} link buoni di oggi, {phish.sum()} phishing di oggi, "
      f"{(h.source=='urlhaus').sum():,} url malware) ===")
print(f"AUC sul fresco: {roc_auc_score(h.label[fresco], p[fresco]):.4f}")
print(pd.DataFrame(righe).to_string(index=False))
cm = confusion_matrix(h.label[fresco], p[fresco] >= .5)
print(f"\na soglia 0,50 -> buoni bloccati {cm[0,1]}/{cm[0].sum()} ({cm[0,1]/cm[0].sum():.1%}), "
      f"phishing preso {cm[1,1]}/{cm[1].sum()} ({cm[1,1]/cm[1].sum():.1%})")

# soglia scelta su meta' dei dati freschi, misurata sull'altra meta': e' quello che si
# farebbe davvero in produzione (si tara su un campione di traffico proprio)
import hashlib
meta = h.registered_domain.map(lambda d: int(hashlib.md5(d.encode()).hexdigest(), 16) % 2 == 0)
A, B = fresco & meta, fresco & ~meta
fa, ta, tha = roc_curve(h.label[A], p[A])
i = np.searchsorted(fa, .01, "right") - 1
soglia = tha[i]
selB = h.label[B] == 0
print("")
print(f"soglia tarata su meta' dell'holdout ({A.sum()} url) = {soglia:.3f}")
print(f"  misurata sull'altra meta' ({B.sum()} url): buoni bloccati "
      f"{(p[B][selB.values] >= soglia).mean():.2%}, phishing preso "
      f"{(p[B][~selB.values] >= soglia).mean():.1%}")

# --- quanto sono affidabili questi numeri: bootstrap sulle due classi separate
def ic(mask, soglia, n=2000, seme=0):
    """Intervallo al 95% per la quota segnalata, ricampionando con reinserimento."""
    v = (p[mask] >= soglia).astype(float)
    if len(v) == 0: return (0.0, 0.0)
    rng = np.random.default_rng(seme)
    campioni = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(campioni, 2.5)), float(np.percentile(campioni, 97.5))

_, _, th_1 = None, None, None
fpr_b, tpr_b, th_b = roc_curve(h.label[fresco], p[fresco])
i = np.searchsorted(fpr_b, .01, "right") - 1
th_1 = th_b[i]
lo_b, hi_b = ic(buoni.values, th_1)
lo_p, hi_p = ic(phish.values, th_1)
print(f"\nalla soglia {th_1:.3f}, con {int(buoni.sum())} buoni e {int(phish.sum())} phishing:")
print(f"  buoni bloccati   {(p[buoni.values] >= th_1).mean():.2%}  (95%: {lo_b:.2%} - {hi_b:.2%})")
print(f"  phishing preso   {(p[phish.values] >= th_1).mean():.1%}  (95%: {lo_p:.1%} - {hi_p:.1%})")
print("  il margine e' largo perche' il phishing verificato di oggi disponibile e' poco:")
print("  e' una stima onesta, non una misura di precisione.")

tr = set(l.strip() for l in open(ROOT / "data/raw/tranco.txt", encoding="utf-8"))
op = h[phish].copy(); op["noto"] = op.registered_domain.isin(tr)
print("\n=== dove sbaglia: phishing su dominio legittimo compromesso vs dominio nuovo ===")
print(op.groupby("noto").agg(url=("p", "size"), preso_a_0_50=("p", lambda s: f"{(s>=.5).mean():.1%}"),
                             punteggio_mediano=("p", lambda s: round(s.median(), 3))).to_string())
json.dump({"ic_buoni": [lo_b, hi_b], "ic_phishing": [lo_p, hi_p],
           "soglia_fpr1": float(th_1), "tabella_interna": tab.round(5).to_dict(), "holdout": righe,
           "auc_holdout": float(roc_auc_score(h.label[fresco], p[fresco])),
           "n_buoni": int(buoni.sum()), "n_phishing": int(phish.sum()),
           "n_urlhaus": int((h.source == "urlhaus").sum())},
          open(OUT / "riepilogo.json", "w"), indent=2)
h.to_csv(OUT / "holdout_scored.csv", index=False)
