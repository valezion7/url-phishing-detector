"""Lead scoring: chi accetta il deposito vincolato dopo una chiamata?
UCI Bank Marketing (41.188 contatti reali, banca portoghese 2008-2010).
Confronta LogReg / Decision Tree / Random Forest / HistGradientBoosting / Voting ensemble.
"""
import json, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_auc_score,
                             average_precision_score, balanced_accuracy_score, f1_score,
                             precision_recall_curve, roc_curve, classification_report)

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
SEED = 42

df = pd.read_csv(ROOT / "data/bank-additional-full.csv", sep=";")
# 'duration' e' la durata della chiamata: si conosce solo DOPO l'esito -> leakage, va via.
df = df.drop(columns=["duration"])
y = (df.pop("y") == "yes").astype(int)
X = df
cat = X.select_dtypes("object").columns.tolist()
num = X.select_dtypes("number").columns.tolist()
assert "duration" not in X.columns, "leakage: duration deve essere rimossa"

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                         ("num", StandardScaler(), num)])

# class_weight ovunque possibile: positivi ~11%
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    "DecisionTree":       DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, class_weight="balanced", random_state=SEED),
    "RandomForest":       RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, random_state=SEED),
}
models["VotingEnsemble"] = VotingClassifier(
    [(k, models[k]) for k in ("LogisticRegression", "RandomForest", "HistGradientBoosting")],
    voting="soft", n_jobs=-1)

def lift_at_k(y_true, score, k=.10):
    """Quanti dei convertiti prendi chiamando solo il top-k% della lista."""
    n = int(len(score) * k)
    idx = np.argsort(score)[::-1][:n]
    return y_true.iloc[idx].sum() / y_true.sum()

results, fitted = {}, {}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    pr, rc, th = precision_recall_curve(yte, p)
    f1s = np.nan_to_num(2 * pr * rc / (pr + rc))
    best_th = float(th[f1s[:-1].argmax()])
    results[name] = {
        "roc_auc": roc_auc_score(yte, p),
        "pr_auc": average_precision_score(yte, p),
        "cv_roc_auc_mean": float(cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1).mean()),
        "balanced_acc@0.5": balanced_accuracy_score(yte, p >= .5),
        "f1@0.5": f1_score(yte, p >= .5),
        "best_threshold": best_th,
        "f1@best": f1_score(yte, p >= best_th),
        "recall_at_top10pct": float(lift_at_k(yte, p)),
        "confusion@0.5": confusion_matrix(yte, p >= .5).tolist(),
    }
    fitted[name] = (pipe, p)
    print(f"{name:22s} ROC-AUC {results[name]['roc_auc']:.3f}  PR-AUC {results[name]['pr_auc']:.3f}  top10% recall {results[name]['recall_at_top10pct']:.1%}")

# --- confusion matrix, una per modello
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (name, (pipe, p)) in zip(axes.ravel(), fitted.items()):
    ConfusionMatrixDisplay(confusion_matrix(yte, p >= .5), display_labels=["no", "si"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nROC-AUC {results[name]['roc_auc']:.3f}")
axes.ravel()[-1].axis("off")
fig.suptitle("Confusion matrix sul test set (8.238 contatti mai visti), soglia 0.50")
fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=130); plt.close(fig)

# --- ROC + PR
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
for name, (pipe, p) in fitted.items():
    fpr, tpr, _ = roc_curve(yte, p); a1.plot(fpr, tpr, label=f"{name} ({results[name]['roc_auc']:.3f})")
    pr, rc, _ = precision_recall_curve(yte, p); a2.plot(rc, pr, label=f"{name} ({results[name]['pr_auc']:.3f})")
a1.plot([0, 1], [0, 1], "k--", lw=.8); a1.set(xlabel="FPR", ylabel="TPR", title="ROC"); a1.legend(fontsize=8)
a2.axhline(yte.mean(), ls="--", c="k", lw=.8); a2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall"); a2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "curves.png", dpi=130); plt.close(fig)

# --- albero leggibile (il modello che il commerciale puo' leggere)
tree_pipe = fitted["DecisionTree"][0]
names = tree_pipe.named_steps["pre"].get_feature_names_out()
fig, ax = plt.subplots(figsize=(22, 9))
plot_tree(tree_pipe.named_steps["clf"], max_depth=3, feature_names=names, class_names=["no", "si"],
          filled=True, impurity=False, proportion=True, fontsize=8, ax=ax)
ax.set_title("Decision tree (primi 3 livelli) - le regole che un umano puo' leggere")
fig.tight_layout(); fig.savefig(OUT / "decision_tree.png", dpi=110); plt.close(fig)

# --- cosa pesa: coefficienti LogReg vs importanze RandomForest
lr = fitted["LogisticRegression"][0].named_steps["clf"].coef_[0]
rf = fitted["RandomForest"][0].named_steps["clf"].feature_importances_
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
i = np.argsort(np.abs(lr))[-15:]
a1.barh(names[i], lr[i], color=np.where(lr[i] > 0, "#2a9d8f", "#e76f51")); a1.set_title("LogReg: top 15 coefficienti")
j = np.argsort(rf)[-15:]
a2.barh(names[j], rf[j], color="#264653"); a2.set_title("RandomForest: top 15 importanze")
fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=130); plt.close(fig)

best = max(results, key=lambda k: results[k]["pr_auc"])
print("\nMigliore per PR-AUC:", best)
print(classification_report(yte, fitted[best][1] >= results[best]["best_threshold"], target_names=["no", "si"], digits=3))

import joblib
joblib.dump(fitted[best][0], OUT / "model.joblib")
json.dump({"dataset": "UCI Bank Marketing (bank-additional-full)", "n_rows": len(df),
           "positive_rate": float(y.mean()), "dropped_leaky": ["duration"], "seed": SEED,
           "best_model": best, "models": results},
          open(OUT / "metrics.json", "w"), indent=2)

# ponytail: self-check minimo - se la pipeline si rompe questi due saltano
assert results[best]["roc_auc"] > .75, "AUC sotto il minimo sensato"
assert results[best]["recall_at_top10pct"] > 3 * .10, "il modello non batte la chiamata a caso"
print("OK ->", OUT)
