"""Controlla un URL (o una lista) con il modello addestrato.

  python check_url.py https://esempio.com/login
  python check_url.py --file lista.txt          -> scrive outputs/controllati.csv

Nessuna chiamata di rete: il giudizio esce dalla sola stringa, quindi si puo' usare
prima di aprire il link.
"""
import json, pathlib, sys
import pandas as pd, joblib
import features as F

ROOT = pathlib.Path(__file__).parent
MOD = ROOT / "outputs_naturale"      # il modello che viene spedito
model = joblib.load(MOD / "model.joblib")
TH = json.load(open(MOD / "thresholds.json"))

MOTIVI = [
    ("brand_foreign_tld", "il nome di un marchio noto sotto un dominio che non e' il suo"),
    ("brand_mismatch", "un marchio noto compare fuori dal dominio registrato"),
    ("is_ip_host", "indirizzo IP al posto del nome"),
    ("is_punycode", "caratteri non latini camuffati (punycode)"),
    ("is_shortener", "accorciatore di link: la destinazione e' nascosta"),
    ("shared_host_deep_sub", "sottodominio su un host dove chiunque puo' pubblicare"),
    ("url_inside_url", "un secondo indirizzo dentro l'indirizzo"),
    ("n_at", "chiocciola nell'URL: nasconde il vero host"),
]

def spiega(f):
    out = [t for k, t in MOTIVI if f.get(k)]
    if f["dom_word_cover"] < .35 and f["dom_len"] >= 7: out.append("nome di dominio che non si legge come parole")
    if f["n_subdomains"] >= 3: out.append(f"{f['n_subdomains']} sottodomini incatenati")
    if f["n_sensitive_words"] >= 2: out.append("parole da pagina di accesso (login/verify/secure...)")
    if f["host_digits"] >= 4: out.append("molte cifre nel nome host")
    return out

def giudizio(p):
    if p >= TH["threshold_fpr01"]: return "PHISHING"
    if p >= TH["threshold_fpr1"]: return "sospetto"
    return "ok"

def score(urls):
    X = F.featurize(urls)
    p = model.predict_proba(X)[:, 1]
    return X, p

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(0)
    if args[0] == "--file":
        urls = [l.strip() for l in open(args[1], encoding="utf-8") if l.strip()]
        X, p = score(urls)
        out = pd.DataFrame({"url": urls, "punteggio": p.round(4),
                            "giudizio": [giudizio(v) for v in p]}).sort_values("punteggio", ascending=False)
        out.to_csv(ROOT / "outputs/controllati.csv", index=False)
        print(out.head(25).to_string(index=False))
        print(f"\n{len(out)} url | PHISHING {(out.giudizio=='PHISHING').sum()} | "
              f"sospetti {(out.giudizio=='sospetto').sum()} -> outputs/controllati.csv")
    else:
        X, p = score(args)
        for u, v, (_, f) in zip(args, p, X.iterrows()):
            print(f"\n{u}\n  punteggio {v:.3f}  ->  {giudizio(v)}")
            for m in spiega(f): print(f"   - {m}")
