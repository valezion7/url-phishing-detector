"""Addestra e, soprattutto, mette alla prova un rilevatore di URL di phishing.

Tre prove, in ordine di severita':
  1. split casuale            -- il numero che si legge di solito nei paper
  2. split per dominio        -- nessun dominio sta sia in train che in test
  3. holdout indipendente     -- phishing verificato di oggi (OpenPhish) contro
                                 link postati oggi su Hacker News, piu' URLhaus
                                 (malware: un'altra minaccia, per vedere se regge)

Punto di lavoro: un filtro che blocca link deve sbagliare pochissimo sui buoni,
quindi la metrica che conta e' il recall a falsi positivi <= 1% e <= 0.1%.
"""
import json, pathlib, sys, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, precision_recall_curve,
                             classification_report, f1_score)
import joblib
try:
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception: pass
import features as F

ROOT = pathlib.Path(__file__).parent
SUFF = "_naturale" if "--naturale" in sys.argv else ""
OUT = ROOT / f"outputs{SUFF}"; OUT.mkdir(exist_ok=True)
SEED, NJOBS = 42, 12
CANDIDATI = {"Ensemble (n-gram + boosting)": "voto", "Stacking (meta-modello)": "stacking"}

t0 = time.time()
data = pd.read_csv(ROOT / f"data/dataset{SUFF}.csv")
if "--prova" in sys.argv:   # giro veloce per verificare che il codice regga
    data = data.sample(int(sys.argv[sys.argv.index("--prova") + 1]), random_state=0).reset_index(drop=True)
    OUT = ROOT / "outputs_prova"; OUT.mkdir(exist_ok=True)
print(f"dataset {len(data):,} url  |  phishing {data.label.mean():.1%}")
X = F.featurize(data.url.tolist())
y = data.label.values; groups = data.registered_domain.values
NUM = [c for c in X.columns if c not in ("tld", "registered_domain", "url_text", "host_text", "path_text")
       and not c.startswith("_")]
print(f"feature estratte in {time.time()-t0:.0f}s: {len(NUM)} numeriche + tld + testo")

def ct_num(dense):
    return ColumnTransformer([("tld", OneHotEncoder(min_frequency=30, handle_unknown="infrequent_if_exist",
                                                    sparse_output=False), ["tld"]),
                              ("num", StandardScaler(), NUM)], sparse_threshold=0 if dense else .3)

def ct_text():
    """Una sola vista: n-grammi di caratteri su tutta la stringa (host + percorso).

    Avevo provato a separarla in tre (host, percorso, parole intere): sul test interno
    andava un filo meglio (0,9849 contro 0,9841) e sull'holdout indipendente crollava
    (0,907 contro 0,950). Le viste separate lasciano memorizzare pezzi di host visti in
    addestramento. Lo split per dominio non se n'era accorto.
    """
    return ColumnTransformer([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=300_000, sublinear_tf=True, lowercase=True), "url_text"),
        ("tld", OneHotEncoder(min_frequency=30, handle_unknown="infrequent_if_exist"), ["tld"]),
        ("num", MaxAbsScaler(), NUM)])

def build():
    p_text = Pipeline([("pre", ct_text()),
                       ("clf", LogisticRegression(C=8, max_iter=3000, solver="liblinear", random_state=SEED))])
    p_hgb = Pipeline([("pre", ct_num(True)),
                      ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=.1,
                                                             max_leaf_nodes=63, early_stopping=True,
                                                             n_iter_no_change=20, random_state=SEED))])
    return {
        "LogisticRegression (feature)": Pipeline([("pre", ct_num(True)), ("clf", LogisticRegression(max_iter=3000, random_state=SEED))]),
        "DecisionTree (feature)": Pipeline([("pre", ct_num(True)), ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=40, random_state=SEED))]),
        "RandomForest (feature)": Pipeline([("pre", ct_num(True)), ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=NJOBS, random_state=SEED))]),
        "HistGradientBoosting (feature)": p_hgb,
        "LogisticRegression (char n-gram)": p_text,
        "Ensemble (n-gram + boosting)": VotingClassifier([("text", p_text), ("hgb", p_hgb)], voting="soft", weights=[1, 1]),
        # stacking: invece di fare la media, una regressione logistica impara quanto
        # fidarsi di ciascuno dei due, usando previsioni fuori campione
        "Stacking (meta-modello)": StackingClassifier([("text", p_text), ("hgb", p_hgb)],
                                                      final_estimator=LogisticRegression(max_iter=1000),
                                                      cv=2, n_jobs=1, passthrough=False),
    }

def recall_at_fpr(yt, p, max_fpr):
    fpr, tpr, th = roc_curve(yt, p)
    i = np.searchsorted(fpr, max_fpr, "right") - 1
    return float(tpr[i]), float(th[i])

def evaluate(yt, p):
    r1, t1 = recall_at_fpr(yt, p, .01); r01, t01 = recall_at_fpr(yt, p, .001)
    cm = confusion_matrix(yt, p >= .5)
    return {"roc_auc": roc_auc_score(yt, p), "pr_auc": average_precision_score(yt, p),
            "accuracy@0.5": float((( p >= .5) == yt).mean()), "f1@0.5": f1_score(yt, p >= .5),
            "recall@fpr1%": r1, "threshold@fpr1%": t1,
            "recall@fpr0.1%": r01, "threshold@fpr0.1%": t01,
            "confusion@0.5": cm.tolist()}

results, preds = {}, {}
rnd_tr, rnd_te = train_test_split(np.arange(len(X)), test_size=.2, stratify=y, random_state=SEED)
dom_tr, dom_te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(X, y, groups))
for split_name, (tr, te) in {"casuale": (rnd_tr, rnd_te), "per_dominio": (dom_tr, dom_te)}.items():
    print(f"\n=== split {split_name}: train {len(tr):,} / test {len(te):,} "
          f"(domini condivisi: {len(set(groups[tr]) & set(groups[te]))})")
    results[split_name] = {}
    for name, model in build().items():
        t = time.time(); model.fit(X.iloc[tr], y[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        m = evaluate(y[te], p); m["fit_s"] = round(time.time() - t, 1)
        results[split_name][name] = m
        preds[(split_name, name)] = (y[te], p)
        print(f"  {name:34s} AUC {m['roc_auc']:.4f}  PR {m['pr_auc']:.4f}  "
              f"acc {m['accuracy@0.5']:.4f}  R@FPR1% {m['recall@fpr1%']:.3f}  R@FPR0.1% {m['recall@fpr0.1%']:.3f}  ({m['fit_s']}s)")
        if split_name == "per_dominio" and name in CANDIDATI:
            joblib.dump(model, OUT / f"model_{CANDIDATI[name]}.joblib")

# il modello da spedire e' quello che va meglio sullo split per dominio, non uno scelto prima
MODELLO_FINALE = max(CANDIDATI, key=lambda k: results["per_dominio"][k]["pr_auc"])
import shutil; shutil.copyfile(OUT / f"model_{CANDIDATI[MODELLO_FINALE]}.joblib", OUT / "model.joblib")
mf = results["per_dominio"][MODELLO_FINALE]
json.dump({"modello": MODELLO_FINALE, "threshold_fpr1": mf["threshold@fpr1%"],
           "threshold_fpr01": mf["threshold@fpr0.1%"]}, open(OUT / "thresholds.json", "w"), indent=2)
print("modello spedito: " + MODELLO_FINALE)
json.dump(results, open(OUT / "metrics_splits.json", "w"), indent=2, default=float)

np.savez_compressed(OUT / "predictions.npz",
                    **{f"{sp}|{nm}|y": v[0] for (sp, nm), v in preds.items()},
                    **{f"{sp}|{nm}|p": v[1] for (sp, nm), v in preds.items()})

# ---------------- figure ----------------
noms = list(build().keys())
fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
for ax in axes.ravel()[len(noms):]: ax.axis("off")
for ax, nm in zip(axes.ravel(), noms):
    yt, pp = preds[("per_dominio", nm)]
    ConfusionMatrixDisplay(confusion_matrix(yt, pp >= .5), display_labels=["buono", "phishing"]).plot(
        ax=ax, colorbar=False, cmap="Blues", values_format=",d")
    mm = results["per_dominio"][nm]
    ax.set_xlabel("giudizio del modello", fontsize=8); ax.set_ylabel("verità", fontsize=8)
    ax.set_title(f"{nm}\nAUC {mm['roc_auc']:.4f} - phishing preso a FPR 1%: {mm['recall@fpr1%']:.1%}", fontsize=9)
fig.suptitle("Confusion matrix, split per dominio (train e test non condividono nessun dominio), soglia 0,50")
fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=130); plt.close(fig)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
for nm in noms:
    yt, pp = preds[("per_dominio", nm)]
    fpr, tpr, _ = roc_curve(yt, pp); a1.plot(fpr, tpr, lw=1.6, label=f"{nm} ({roc_auc_score(yt,pp):.4f})")
    pr, rc, _ = precision_recall_curve(yt, pp); a2.plot(rc, pr, lw=1.6, label=f"{nm} ({average_precision_score(yt,pp):.4f})")
a1.set(xlabel="quota di buoni bloccati", ylabel="quota di phishing preso", title="ROC (scala log)", xscale="log", xlim=(1e-4, 1))
a2.set(xlabel="recall", ylabel="precisione", title="Precision-Recall")
for a in (a1, a2): a.legend(fontsize=7.5); a.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT / "curves.png", dpi=130); plt.close(fig)

yt, pp = preds[("per_dominio", MODELLO_FINALE)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5))
a1.hist(pp[yt == 0], bins=60, alpha=.8, label="buoni", color="#2a7f62")
a1.hist(pp[yt == 1], bins=60, alpha=.8, label="phishing", color="#b23a2e")
a1.set(xlabel="punteggio", ylabel="url (scala log)", title="Separazione dei punteggi", yscale="log"); a1.legend()
fpr, tpr, th = roc_curve(yt, pp)
a2.plot(fpr, tpr, lw=2, color="#22508f")
for f in (.001, .01, .05):
    i = np.searchsorted(fpr, f, "right") - 1
    a2.plot(fpr[i], tpr[i], "o", color="#b23a2e")
    a2.annotate(f"blocco {f:.1%} dei buoni\nprendo {tpr[i]:.1%} del phishing\nsoglia {th[i]:.2f}",
                (fpr[i], tpr[i]), textcoords="offset points", xytext=(12, -26), fontsize=8)
a2.set(xscale="log", xlim=(1e-4, 1), xlabel="quota di buoni bloccati", ylabel="quota di phishing preso",
       title="Dove mettere la soglia"); a2.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT / "soglia.png", dpi=130); plt.close(fig)

alb = Pipeline([("pre", ct_num(True)),
                ("clf", DecisionTreeClassifier(max_depth=4, min_samples_leaf=300, random_state=SEED))]).fit(X.iloc[dom_tr], y[dom_tr])
fig, ax = plt.subplots(figsize=(21, 8.5))
plot_tree(alb.named_steps["clf"], max_depth=3, feature_names=alb.named_steps["pre"].get_feature_names_out(),
          class_names=["buono", "phishing"], filled=True, impurity=False, proportion=True, fontsize=8, ax=ax)
ax.set_title("Le prime domande che si pone un albero decisionale")
fig.tight_layout(); fig.savefig(OUT / "albero.png", dpi=110); plt.close(fig)

txt = build()["LogisticRegression (char n-gram)"].fit(X.iloc[dom_tr], y[dom_tr])
fn = txt.named_steps["pre"].get_feature_names_out(); co = txt.named_steps["clf"].coef_[0]
ETICHETTA = {"host": "host", "path": "percorso", "parole": "parola"}
testuali = [i for i, n in enumerate(fn) if n.split("__", 1)[0] in ETICHETTA]
numeriche = [i for i, n in enumerate(fn) if n.startswith("num__")]

def nome(i):
    fam, resto = fn[i].split("__", 1)
    return f"{ETICHETTA[fam]}  «{resto}»" if fam in ETICHETTA else resto

fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.8))
for ax, idx, titolo, col in (
        (axes[0], sorted(testuali, key=lambda i: -co[i])[:18], "Pezzi di stringa che accusano", "#b23a2e"),
        (axes[1], sorted(testuali, key=lambda i: co[i])[:18], "Pezzi di stringa che scagionano", "#2a7f62")):
    idx = idx[::-1]
    ax.barh([nome(i) for i in idx], [co[i] for i in idx], color=col)
    ax.set_title(titolo, fontsize=11)
idx = sorted(numeriche, key=lambda i: -abs(co[i]))[:18][::-1]
axes[2].barh([nome(i) for i in idx], [co[i] for i in idx],
             color=["#b23a2e" if co[i] > 0 else "#2a7f62" for i in idx])
axes[2].set_title("Feature calcolate, peso con segno" + chr(10) + "(rosso = verso phishing)", fontsize=11)
for a in axes: a.tick_params(labelsize=8); a.grid(alpha=.2, axis="x"); a.axvline(0, c="#333", lw=.8)
fig.suptitle("Che cosa guarda la regressione logistica sui caratteri (split per dominio)")
fig.tight_layout(); fig.savefig(OUT / "ngrammi.png", dpi=130); plt.close(fig)
print(f"\ntotale {time.time()-t0:.0f}s -> {OUT}")
