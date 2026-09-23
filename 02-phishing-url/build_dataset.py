"""Da liste grezze a dataset onesto.

Ogni regola qui sotto nasce da un errore vero visto in fase di prova:
 - max 3 URL per dominio registrato: le liste di phishing hanno migliaia di pagine
   sullo stesso host compromesso; senza tetto il modello impara quegli host.
 - homepage campionate su TUTTA la classifica Tranco (non solo in cima): alla prima
   prova il modello bloccava i siti piccoli perche' fra i buoni c'erano solo marchi noti.
 - quote per TLD: nei primi dati i benigni non avevano NEMMENO UN .ai/.io/.dev mentre
   il phishing aveva .top/.xyz/.tk, quindi bastava il TLD per indovinare. Ora ogni TLD
   usato dal phishing ha anche dei benigni veri.
 - benigni di tre forme (homepage, pagina profonda citata da Wikipedia, pagina da
   sitemap di siti veri) perche' il phishing ha entrambe le forme.
 - URLhaus (malware) e OpenPhish/HackerNews (di oggi) restano FUORI dal training.
"""
import math, pathlib, random, sys
import pandas as pd
try:
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception: pass
import tldextract

ROOT = pathlib.Path(__file__).parent
RAW = ROOT / "data/raw"
ext = tldextract.TLDExtract(cache_dir=str(ROOT / ".tld_cache"))
random.seed(42)
CAP_PHISH, CAP_BENIGN = 3, 5
# due modi di comporre i benigni, si confrontano sull'holdout indipendente:
#   pareggiato -> per ogni (TLD, forma) tanti buoni quanti cattivi: nessuna scorciatoia,
#                 ma butta via anche il segnale vero (i TLD abusati lo sono davvero)
#   naturale   -> proporzioni del web reale, garantendo pero' che OGNI TLD usato dal
#                 phishing abbia anche dei benigni (era quello che mancava e faceva danni)
MODO = "naturale" if "--naturale" in sys.argv else "pareggiato"
SUFF = "_naturale" if MODO == "naturale" else ""
print("modo:", MODO)

def load(name):
    f = RAW / name
    return [l.strip() for l in f.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()] if f.exists() else []

def norm(urls, source):
    rows, seen = [], set()
    for u in urls:
        if u.startswith(("ftp://", "ftps://")) or " " in u: continue
        if "://" not in u: u = "http://" + u
        host = u.split("://", 1)[1].split("/")[0].split("@")[-1].split(":")[0].lower()
        if host.startswith("www."): host = host[4:]
        if "." not in host or len(host) < 4: continue
        key = u.split("://", 1)[1].lstrip("www.").rstrip("/")
        if key in seen: continue
        seen.add(key)
        e = ext(host)
        rows.append((u, ".".join(p for p in (e.domain, e.suffix) if p) or host, e.suffix, source))
    return pd.DataFrame(rows, columns=["url", "registered_domain", "tld", "source"])

def cap(df, n):
    return df.groupby("registered_domain", group_keys=False, sort=False).head(n)

def tranco_index():
    """domini Tranco con il loro TLD, calcolato una volta sola."""
    f = RAW / "tranco_tld.csv"
    if f.exists(): return pd.read_csv(f)
    doms = load("tranco.txt")
    df = pd.DataFrame({"domain": doms, "rank": range(1, len(doms) + 1)})
    df["tld"] = [ext(d).suffix for d in doms]
    df.to_csv(f, index=False); return df

print("carico liste...")
phish = cap(norm(load("phishing_db.txt"), "phishing_db"), CAP_PHISH)
wtld = cap(norm(load("wikipedia_tld.txt"), "wikipedia_tld"), CAP_BENIGN)
wtld = wtld.groupby("tld", group_keys=False, sort=False).head(3500)   # nessun TLD deve dominare
deep = pd.concat([cap(norm(load("wikipedia.txt"), "wikipedia"), CAP_BENIGN),
                  cap(norm(load("sitemaps.txt"), "sitemaps"), CAP_BENIGN),
                  cap(norm(load("sitemaps_modern.txt"), "sitemaps_modern"), CAP_BENIGN),
                  cap(norm(load("pagine_accesso.txt") + load("pagine_accesso2.txt"), "pagine_accesso"), CAP_BENIGN),
                  cap(norm(load("wikipedia_lingue.txt"), "wikipedia_lingue"), CAP_BENIGN),
                  wtld], ignore_index=True).drop_duplicates("url")
deep = deep[~deep.registered_domain.isin(set(phish.registered_domain))]
print(f"  phishing disponibili {len(phish):,} | pagine profonde benigne {len(deep):,}")

# --- pareggio per TLD e per forma --------------------------------------------
# Per ogni combinazione (TLD, ha un path si/no) prendo tanti benigni quanti phishing.
# Cosi' ne' il TLD ne' la forma dell'URL dicono da soli la risposta: sono le due
# scorciatoie che il modello si era preso nelle prime versioni (bloccava ogni .xyz,
# e considerava phishing qualsiasi cosa avesse un percorso dopo il dominio).
tr = tranco_index()
tr = tr[~tr.domain.isin(set(deep.registered_domain) | set(phish.registered_domain))]
home_by_tld = {t: g for t, g in tr.groupby("tld")}

def has_path(df):
    return df.url.str.split("://").str[1].str.rstrip("/").str.contains("/")

deep, phish = deep.copy(), phish.copy()
for d in (deep, phish): d["forma"] = has_path(d).map({True: "profondo", False: "radice"})

if MODO == "pareggiato":
    righe_b, righe_p, scarsi = [], [], []
    for (tld, forma), gp in phish.groupby(["tld", "forma"], sort=False):
        gb = deep[(deep.tld == tld) & (deep.forma == forma)]
        if forma == "radice" and len(gb) < len(gp) and tld in home_by_tld:
            extra = home_by_tld[tld].sample(min(len(home_by_tld[tld]), len(gp) - len(gb)), random_state=42)
            gb = pd.concat([gb, norm([f"https://{d}" for d in extra.domain], "tranco_home").assign(forma="radice")])
        n = min(len(gb), len(gp))
        if n == 0:
            scarsi.append(f"{tld}/{forma}:{len(gp)}"); continue
        righe_b.append(gb.head(n)); righe_p.append(gp.sample(n, random_state=42))
    benign = pd.concat(righe_b, ignore_index=True)
    phish = pd.concat(righe_p, ignore_index=True)
    persi = sum(int(x.split(":")[1]) for x in scarsi)
    print(f"  pareggio (TLD x forma): {len(benign):,} per classe | celle senza benigni: {len(scarsi)} ({persi:,} phishing scartati)")
else:
    # copertura garantita: ogni TLD del phishing deve avere dei benigni veri, poi
    # si riempie con homepage fino a una quota di radici simile a quella del phishing
    quote, mancanti = [], []
    for tld, n in phish.tld.value_counts().items():
        avail_deep = len(deep[deep.tld == tld])
        need = max(30, int(n * .25)) - avail_deep
        h = home_by_tld.get(tld)
        if need > 0 and h is not None and len(h):
            quote.append(h.sample(min(len(h), need), random_state=42))
        elif need > 0: mancanti.append(tld)
    coperture = norm([f"https://{d}" for d in pd.concat(quote).domain], "tranco_home") if quote else deep.head(0)
    quota_radici = int(len(deep) / max(.01, 1 - (phish.forma == "radice").mean())) - len(deep)
    resto = tr[~tr.domain.isin(set(coperture.registered_domain))].sample(
        min(len(tr), max(0, quota_radici - len(coperture))), random_state=42)
    benign = pd.concat([deep, coperture, norm([f"https://{d}" for d in resto.domain], "tranco_home")],
                       ignore_index=True).drop_duplicates("url")
    # tetto: oltre questa soglia il guadagno e' marginale e l'addestramento si allunga molto
    n = min(len(benign), len(phish), 130_000)
    # Le pagine di accesso vere sono poche (~1.600) ma sono LA categoria che il phishing
    # imita: lasciate alla loro frequenza naturale sparirebbero nel campione (l'1%) e il
    # modello continuerebbe a bollare come truffa il login di chiunque. Qui entrano tutte,
    # ripetute, fino al 7% dei benigni.
    acc = benign[benign.source == "pagine_accesso"].drop_duplicates("url")
    resto = benign[benign.source != "pagine_accesso"]
    if len(acc):
        quota = min(len(acc) * 6, int(n * .07))
        acc_rip = pd.concat([acc] * 6, ignore_index=True).head(quota)
        benign = pd.concat([acc_rip, resto.sample(min(len(resto), n - len(acc_rip)), random_state=42)],
                           ignore_index=True)
        print(f"  pagine di accesso: {len(acc):,} distinte -> {len(acc_rip):,} righe "
              f"({len(acc_rip)/n:.1%} dei benigni)")
    else:
        benign = benign.sample(n, random_state=42)
    phish = phish.sample(n, random_state=42)
    print(f"  copertura garantita: {len(benign):,} per classe | TLD senza benigni disponibili: {len(mancanti)}")
    for d in (benign,): d["forma"] = has_path(d).map({True: "profondo", False: "radice"})

# niente drop_duplicates qui: le ripetizioni delle pagine di accesso sono volute
data = pd.concat([phish.assign(label=1), benign.assign(label=0)],
                 ignore_index=True).sample(frac=1, random_state=42)

# --- controllo scorciatoie: le due classi devono assomigliarsi nella forma
path = data.url.str.split("://").str[1].str.rstrip("/").str.contains("/")
rep = pd.DataFrame({
    "n": data.groupby("label").size(),
    "domini": data.groupby("label").registered_domain.nunique(),
    "https_%": (data.url.str.startswith("https").groupby(data.label).mean() * 100).round(1),
    "con_path_%": (path.groupby(data.label).mean() * 100).round(1),
    "len_mediana": data.url.str.len().groupby(data.label).median(),
    "tld_diversi": data.groupby("label").tld.nunique(),
})
print("\n--- forma delle due classi\n" + rep.to_string())
top = data.tld.value_counts().head(14).index
share = (pd.crosstab(data.tld, data.label, normalize="columns").loc[top] * 100).round(2)
share.columns = ["benigni_%", "phishing_%"]
print("\n--- TLD piu' frequenti (nessuno deve essere di una classe sola)\n" + share.to_string())
print("\nfonti:", data.groupby(["label", "source"]).size().to_dict())

data.to_csv(ROOT / f"data/dataset{SUFF}.csv", index=False)
holdout = pd.concat([norm(load("openphish.txt"), "openphish").assign(label=1),
                     norm(load("hackernews.txt"), "hackernews").assign(label=0),
                     norm(load("lobsters.txt"), "lobsters").assign(label=0),
                     norm(load("hackernews_recenti.txt"), "hackernews_recenti").assign(label=0),
                     norm(load("urlhaus.txt"), "urlhaus").assign(label=1)], ignore_index=True)
holdout = holdout[~holdout.registered_domain.isin(set(data.registered_domain))]
holdout.to_csv(ROOT / "data/holdout_fresh.csv", index=False)
print(f"\ndataset {len(data):,} -> data/dataset.csv | holdout {len(holdout):,}", holdout.groupby("source").size().to_dict())
