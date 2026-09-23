"""Demo web del rilevatore di phishing / web demo of the phishing detector.

Gira su Hugging Face Spaces (CPU gratuita) o in locale:
    pip install -r requirements.txt && python app.py

Il modello viene scaricato dal repository del modello su Hugging Face la prima volta.
Se trovi `modello/` accanto a questo file, usa quello (utile in locale).
"""
import json, os, pathlib, sys
import gradio as gr

QUI = pathlib.Path(__file__).parent
REPO_MODELLO = os.environ.get("MODEL_REPO", "valezion/url-phishing-detector")

def carica():
    locale = QUI / "modello"
    if (locale / "model.joblib").exists():
        base = locale
    else:
        from huggingface_hub import snapshot_download
        base = pathlib.Path(snapshot_download(repo_id=REPO_MODELLO))
    sys.path.insert(0, str(base))
    import joblib, features as F
    F.RAW = base / "data/raw"                 # non c'e': usa il vocabolario compatto
    F.DERIVATI = base / "vocabolario"
    soglie = json.loads((base / "thresholds.json").read_text(encoding="utf-8"))
    return joblib.load(base / "model.joblib"), F, soglie

MODELLO, F, SOGLIE = carica()
TH_SOSPETTO, TH_PHISHING = SOGLIE["threshold_fpr1"], SOGLIE["threshold_fpr01"]

MOTIVI = [
    ("brand_foreign_tld", "il nome di un marchio noto sotto un dominio che non è il suo",
                          "a known brand's name under a domain that is not its own"),
    ("brand_mismatch", "un marchio noto compare nel nome host ma il dominio registrato è un altro",
                       "a known brand appears in the hostname but the registered domain is another"),
    ("is_ip_host", "indirizzo IP al posto del nome", "an IP address instead of a name"),
    ("is_punycode", "caratteri non latini camuffati (punycode)", "disguised non-Latin characters (punycode)"),
    ("is_shortener", "accorciatore di link: la destinazione è nascosta",
                     "a link shortener: the destination is hidden"),
    ("shared_host_deep_sub", "sottodominio su un host dove chiunque può pubblicare",
                             "a subdomain on a host where anyone can publish"),
    ("url_inside_url", "un secondo indirizzo dentro l'indirizzo", "a second address inside the address"),
    ("n_at", "chiocciola nell'URL: nasconde il vero host", "an @ in the URL: it hides the real host"),
]

def spiega(f):
    it, en = [], []
    for k, i, e in MOTIVI:
        if f.get(k):
            it.append(i); en.append(e)
    if f["dom_word_cover"] < .35 and f["dom_len"] >= 7:
        it.append("nome di dominio che non si legge come parole")
        en.append("a domain name that does not read as words")
    if f["n_subdomains"] >= 3:
        it.append(f"{f['n_subdomains']} sottodomini incatenati")
        en.append(f"{f['n_subdomains']} chained subdomains")
    if f["n_sensitive_words"] >= 2:
        it.append("parole da pagina di accesso (login, verify, accedi, verifica...)")
        en.append("login-page words (login, verify, accedi, verifica...)")
    if f["host_digits"] >= 4:
        it.append("molte cifre nel nome host"); en.append("many digits in the hostname")
    return it, en

def giudica(url):
    url = (url or "").strip()
    if not url:
        return "", ""
    X = F.featurize([url])
    p = float(MODELLO.predict_proba(X)[0, 1])
    f = X.iloc[0]
    it, en = spiega(f)
    if p >= TH_PHISHING:
        tit, col = "PHISHING", "#b23a2e"
    elif p >= TH_SOSPETTO:
        tit, col = "SOSPETTO / SUSPICIOUS", "#a9761a"
    else:
        tit, col = "NESSUN SEGNALE / NO SIGNAL", "#0d6b55"
    testa = (f"<div style='font-family:system-ui;border-left:4px solid {col};padding:10px 16px'>"
             f"<div style='font-size:1.5rem;font-weight:700;color:{col}'>{tit}</div>"
             f"<div style='font-family:ui-monospace,monospace;font-size:.9rem;opacity:.75'>"
             f"punteggio / score {p:.3f} &nbsp;·&nbsp; soglie {TH_SOSPETTO:.2f} / {TH_PHISHING:.2f}</div></div>")
    if it:
        corpo = ("**Che cosa ha notato / what it noticed**\n\n"
                 + "\n".join(f"- {a}  \n  *{b}*" for a, b in zip(it, en)))
    else:
        corpo = ("*Nessuno dei segnali espliciti è scattato: il punteggio viene dagli n-grammi "
                 "dei caratteri.*  \n*None of the explicit signals fired: the score comes from the "
                 "character n-grams.*")
    return testa, corpo

ESEMPI = [
    ["https://www.poste.it/"],
    ["http://paypal.com.secure-login.verify-account.tk/webscr?cmd=_login"],
    ["https://github.com/scikit-learn/scikit-learn/pull/28123"],
    ["https://intesasanpaolo.it-sicurezza-clienti.top/accesso/verifica.php?id=8812"],
    ["https://un-scrabbled.com"],
    ["http://192.168.4.11:8080/wp-content/dhl/tracking.php"],
]

INTRO = """
# Is this link phishing? / Questo link è phishing?

Judged from **the string alone** — no network call, no page fetch. Type or paste a URL.

Giudicato dalla **sola stringa** — nessuna chiamata di rete, nessun fetch della pagina.

> **This is a research demo, not a security product.** On an independently collected
> holdout it catches about half of fresh phishing while blocking ~1% of good links, and it
> is blind to scams hosted inside legitimate compromised sites. A "no signal" answer is not
> a guarantee that a link is safe.
>
> **Demo di ricerca, non un prodotto di sicurezza.** Su dati raccolti da altri riconosce
> circa metà del phishing fresco bloccando l'1% dei link buoni, ed è cieco davanti alle
> truffe ospitate dentro siti legittimi compromessi. "Nessun segnale" non è una garanzia.

Code, data sources and the diary of eight ways this model fooled me:
**[github.com/valezion7/url-phishing-detector](https://github.com/valezion7/url-phishing-detector)**
"""

with gr.Blocks(title="Phishing URL detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        casella = gr.Textbox(label="URL", placeholder="https://...", scale=5, max_lines=1)
        bottone = gr.Button("Controlla / Check", variant="primary", scale=1)
    verdetto = gr.HTML()
    dettaglio = gr.Markdown()
    gr.Examples(ESEMPI, inputs=casella, label="Esempi / examples (none of these is an invitation to visit them)")
    for evento in (bottone.click, casella.submit):
        evento(giudica, inputs=casella, outputs=[verdetto, dettaglio])

if __name__ == "__main__":
    demo.launch()
