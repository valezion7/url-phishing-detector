"""Controllo onesto: split cronologico invece che casuale.
Le righe del dataset sono in ordine di tempo (mag 2008 -> nov 2010) e tre feature
(euribor3m, nr.employed, emp.var.rate) sono di fatto un orologio: con lo split casuale
il modello vede il futuro. Qui alleno sul primo 80% e testo sull'ultimo 20%.
"""
import json, pathlib
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

ROOT = pathlib.Path(__file__).parent
df = pd.read_csv(ROOT / "data/bank-additional-full.csv", sep=";").drop(columns=["duration"])
y = (df.pop("y") == "yes").astype(int); X = df
cut = int(len(X) * .8)
MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

def run(cols, label):
    Xc = X[cols]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), Xc.select_dtypes("object").columns.tolist()),
                             ("num", StandardScaler(), Xc.select_dtypes("number").columns.tolist())], sparse_threshold=0)
    out = {}
    for name, clf in [("LogisticRegression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
                      ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, random_state=42))]:
        p = Pipeline([("pre", pre), ("clf", clf)]).fit(Xc[:cut], y[:cut]).predict_proba(Xc[cut:])[:, 1]
        k = int(len(p) * .10); top = np.argsort(p)[::-1][:k]
        out[name] = {"roc_auc": roc_auc_score(y[cut:], p), "pr_auc": average_precision_score(y[cut:], p),
                     "recall_at_top10pct": float(y[cut:].iloc[top].sum() / y[cut:].sum()),
                     "confusion@0.5": confusion_matrix(y[cut:], p >= .5).tolist()}
        print(f"[{label}] {name:22s} ROC-AUC {out[name]['roc_auc']:.3f}  PR-AUC {out[name]['pr_auc']:.3f}  top10% {out[name]['recall_at_top10pct']:.1%}")
    return out

res = {"train_rows": cut, "test_rows": len(X) - cut, "test_positive_rate": float(y[cut:].mean()),
       "con_macro": run(X.columns.tolist(), "con macro"),
       "senza_macro": run([c for c in X.columns if c not in MACRO], "senza macro")}
json.dump(res, open(ROOT / "outputs/metrics_time_split.json", "w"), indent=2)
assert res["senza_macro"]["HistGradientBoosting"]["recall_at_top10pct"] > .10, "peggio del caso: qualcosa non torna"
print("test positivi:", f"{y[cut:].mean():.1%}", "| OK -> outputs/metrics_time_split.json")
