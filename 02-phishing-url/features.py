"""Feature estratte dalla sola stringa dell'URL: nessuna chiamata di rete, nessun
fetch della pagina. Cosi' il classificatore puo' girare prima che l'utente clicchi.

Le 'brand' non sono una lista scritta a mano: sono i nomi di dominio piu' popolari
secondo Tranco. Il segnale che conta e' "un marchio famoso compare nell'URL ma il
dominio registrato non e' il suo" -- la meccanica del phishing, in una feature.
"""
import math, pathlib, re
from collections import Counter
from functools import lru_cache
from urllib.parse import urlsplit, unquote

import pandas as pd
import tldextract

RAW = pathlib.Path(__file__).parent / "data/raw"
_ext = tldextract.TLDExtract(cache_dir=str(pathlib.Path(__file__).parent / ".tld_cache"))

IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
HEX_RE = re.compile(r"^[0-9a-f]{16,}$", re.I)
B64_RE = re.compile(r"[A-Za-z0-9+/=]{24,}")
TOKEN_RE = re.compile(r"[^a-z0-9]+")
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
              "cutt.ly", "rb.gy", "shorturl.at", "rebrand.ly", "t.ly", "bit.do", "u.to",
              "tiny.cc", "shorte.st", "adf.ly", "clck.ru", "v.gd", "s.id", "lnkd.in",
              "short.io", "urlz.fr", "qr.ae", "1url.com", "soo.gd", "bc.vc"}
# Host dove chiunque puo' aprire un sottodominio in un minuto: il dominio registrato
# non dice niente su chi c'e' dentro. Vale per i buoni (github.io) e per i cattivi
# (duckdns.org): la feature dice al modello "qui guarda il sottodominio, non il dominio".
SHARED_HOSTS = {"duckdns.org", "no-ip.org", "no-ip.com", "ddns.net", "hopto.org", "zapto.org",
                "serveo.net", "ngrok.io", "ngrok-free.app", "trycloudflare.com", "loca.lt",
                "000webhostapp.com", "weebly.com", "wixsite.com", "web.app", "firebaseapp.com",
                "github.io", "gitlab.io", "netlify.app", "vercel.app", "pages.dev", "workers.dev",
                "herokuapp.com", "glitch.me", "repl.co", "blogspot.com", "wordpress.com",
                "medium.com", "substack.com", "tumblr.com", "sites.google.com", "r2.dev",
                "azurewebsites.net", "cloudfront.net", "s3.amazonaws.com", "myshopify.com"}
# Parole da pagina di accesso, non solo in inglese: il phishing parla la lingua della
# vittima, e i feed contengono campagne italiane, spagnole, francesi e tedesche.
SENSITIVE = ("login", "signin", "log-in", "verify", "verification", "secure", "account",
             "update", "confirm", "password", "webscr", "banking", "wallet", "invoice",
             "billing", "recover", "unlock", "suspend", "authenticate", "session",
             "accedi", "accesso", "verifica", "sicurezza", "pagamento", "conferma",
             "fattura", "rinnovo", "scadenza", "sospeso", "aggiorna", "bancario",
             "acceso", "ingresar", "verificar", "seguridad", "cuenta", "pago", "factura",
             "connexion", "identifiant", "securite", "compte", "paiement", "facture",
             "anmelden", "konto", "sicherheit", "zahlung", "rechnung", "bestaetigen")

@lru_cache(maxsize=1)
def known_domains(n=120_000):
    """I domini registrati piu' visitati: servono solo per capire se un marchio sta
    a casa propria. paypal.com c'e', paypal.com.security-check.xyz no."""
    f = RAW / "tranco.txt"
    if not f.exists(): return frozenset()
    return frozenset(f.read_text(encoding="utf-8").splitlines()[:n])

@lru_cache(maxsize=1)
def brands(n=5000):
    """Nomi di secondo livello dei domini piu' visitati al mondo (Tranco)."""
    f = RAW / "tranco.txt"
    if not f.exists(): return frozenset()
    words, _, _ = _lex()
    out, top = set(), set()
    for i, d in enumerate(f.read_text(encoding="utf-8").splitlines()[:n]):
        lbl = _ext(d).domain
        if len(lbl) < 4: continue
        if i < 300: top.add(lbl)          # i primissimi valgono anche se sono parole (apple, amazon)
        elif lbl not in words: out.add(lbl)   # piu' giu' tengo solo i nomi non-parola: "code" o
    return frozenset(out | top)               # "prose" sono domini veri ma non sono marchi imitati

# --- leggibilita' del nome a dominio ------------------------------------------
# I falsi positivi peggiori erano homepage di siti piccoli ma dal nome parlante
# (un-scrabbled.com, proudsend.com) scambiate per phishing; i phishing mancati erano
# domini di lettere a caso (wbeuvvfx.com). La differenza non e' la lunghezza: e' se
# il nome si legge. Queste due feature misurano proprio quello.
WORD_RE = re.compile(r"[a-z]+|\d+")

@lru_cache(maxsize=1)
def _lex():
    f = RAW / "words_alpha.txt"
    words = {w for w in f.read_text(encoding="utf-8").split() if len(w) >= 3} if f.exists() else set()
    words |= {"app", "web", "dev", "api", "io", "ai", "3d", "hub", "lab", "net", "biz"}
    bg = Counter(); uni = Counter()
    for w in words:
        t = "^" + w + "$"
        for a, b in zip(t, t[1:]): bg[a + b] += 1; uni[a] += 1
    return frozenset(words), bg, uni

@lru_cache(maxsize=200_000)
def _chunk_cover(chunk):
    words, _, _ = _lex()
    if chunk.isdigit(): return len(chunk)
    i = cov = 0
    while i < len(chunk):
        for j in range(min(len(chunk), i + 18), i + 2, -1):
            if chunk[i:j] in words: cov += j - i; i = j; break
        else: i += 1
    return cov

def word_cover(s, maxlen=120):
    """Quanta parte del nome e' fatta di parole vere (segmentazione greedy).
    Taglio a 120 caratteri: oltre non aggiunge segnale e costerebbe caro su ogni URL."""
    if not _lex()[0]: return 0.0
    tot = cov = 0
    for chunk in WORD_RE.findall(s[:maxlen]):
        tot += len(chunk); cov += _chunk_cover(chunk)
    return cov / max(1, tot)

@lru_cache(maxsize=200_000)
def bigram_lp(s):
    """Quanto il nome 'suona' come una parola: log-prob media dei bigrammi."""
    s = s[:120]
    _, bg, uni = _lex()
    if not bg: return 0.0
    s = "".join(c for c in s.lower() if c.isalpha())
    if len(s) < 3: return 0.0
    t = "^" + s + "$"; tot = 0.0
    for a, b in zip(t, t[1:]):
        tot += math.log2((bg[a + b] + .5) / (uni[a] + 30))
    return tot / (len(t) - 1)

def consonant_run(s):
    best = cur = 0
    for c in s.lower():
        cur = cur + 1 if c.isalpha() and c not in "aeiouy" else 0
        best = max(best, cur)
    return best

def entropy(s):
    if not s: return 0.0
    c = Counter(s); n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())

def extract(url):
    url = url.strip()
    if "://" not in url: url = "http://" + url
    sp = urlsplit(url)
    try:                       # una stringa storta non deve far cadere il chiamante
        host = (sp.hostname or "").lower()
    except ValueError:
        host = ""
    try:
        porta = sp.port
    except ValueError:
        porta = None
    # Canonicalizzazione: lo slash finale della home e il prefisso www. non dicono nulla
    # sulla pericolosita' ma dicono da QUALE LISTA viene l'URL (le homepage Tranco
    # finivano tutte con "/", quelle di Hacker News no). Senza questo il modello impara
    # la provenienza del dato invece del phishing -- e' successo davvero, alla prima prova.
    if host.startswith("www."): host = host[4:]
    path, query, frag = sp.path, sp.query, sp.fragment
    if path in ("", "/") and not query and not frag: path = ""
    rest = path + ("?" + query if query else "") + ("#" + frag if frag else "")
    e = _ext(host)
    reg = ".".join(p for p in (e.domain, e.suffix) if p)
    labels = [l for l in host.split(".") if l]
    sub = host[: -len(reg)].rstrip(".") if reg and host.endswith(reg) else ""
    low = url.lower()
    decoded = unquote(low)
    toks = [t for t in TOKEN_RE.split(low) if t]

    # Il marchio conta se sta nel NOME HOST: e' li' che il phishing lo mette per
    # sembrare autentico. Nel percorso e' un segnale molto piu' debole e pieno di falsi
    # allarmi: "?service=mail" faceva scattare "mail" (mail.ru) su accounts.google.com.
    brand_host = brands() & set(TOKEN_RE.split(sub))
    brand_path = brands() & set(TOKEN_RE.split(rest.lower()))
    f = {
        # forma generale
        "url_len": len(url), "host_len": len(host), "path_len": len(path),
        "query_len": len(query), "n_params": query.count("=") if query else 0,
        "has_fragment": int(bool(frag)), "_is_https": int(sp.scheme == "https"),
        "has_port": int(bool(porta)), "path_depth": path.count("/"),
        # host
        "is_ip_host": int(bool(IP_RE.match(host))),
        "n_subdomains": max(0, len(labels) - len(reg.split(".")) if reg else len(labels)),
        "sub_len": len(sub), "_has_www": int((sp.hostname or "").lower().startswith("www.")),
        "is_punycode": int("xn--" in host), "host_hyphens": host.count("-"),
        "host_digits": sum(c.isdigit() for c in host),
        "host_digit_ratio": sum(c.isdigit() for c in host) / max(1, len(host)),
        "host_entropy": entropy(host), "n_dots": host.count("."),
        "longest_label": max((len(l) for l in labels), default=0),
        "tld_len": len(e.suffix), "is_shortener": int(reg in SHORTENERS),
        # marchi
        "brand_in_url": int(bool(brand_host or brand_path)),
        "brand_mismatch": int(bool(brand_host) and e.domain not in brand_host),
        "brand_in_subdomain": int(bool(brand_host)),
        "brand_in_path": int(bool(brand_path) and e.domain not in brand_path),
        # roblox.com.do, paypal.com.br.verify.xyz: il nome e' quello giusto, la casa no
        "brand_foreign_tld": int(e.domain in brands() and reg not in known_domains()),
        "is_shared_host": int(reg in SHARED_HOSTS),
        "shared_host_deep_sub": int(reg in SHARED_HOSTS and sub.count(".") >= 1),
        # caratteri sospetti
        "n_hyphen": url.count("-"), "n_underscore": url.count("_"), "n_at": url.count("@"),
        "n_tilde": url.count("~"), "n_percent": url.count("%"), "n_plus": url.count("+"),
        "n_equals": url.count("="), "n_amp": url.count("&"), "n_qmark": url.count("?"),
        "n_digits": sum(c.isdigit() for c in url),
        "digit_ratio": sum(c.isdigit() for c in url) / max(1, len(url)),
        "upper_ratio": sum(c.isupper() for c in url) / max(1, len(url)),
        "url_entropy": entropy(url),
        "double_slash_in_path": int("//" in path),
        "url_inside_url": int(low.count("http") > 1),
        "encoded_chars": int(decoded != low),
        # token
        "n_tokens": len(toks), "longest_token": max((len(t) for t in toks), default=0),
        "mean_token_len": sum(map(len, toks)) / max(1, len(toks)),
        "has_hex_token": int(any(HEX_RE.match(t) for t in toks)),
        "has_b64_blob": int(bool(B64_RE.search(rest))),
        "max_token_entropy": max((entropy(t) for t in toks if len(t) >= 8), default=0.0),
        "n_sensitive_words": sum(w in decoded for w in SENSITIVE),
        "ends_executable": int(bool(re.search(r"\.(exe|apk|zip|scr|js|php|htm|html)$", path, re.I))),
        # contesto
        # leggibilita'
        "dom_len": len(e.domain), "dom_word_cover": word_cover(e.domain),
        "dom_bigram_lp": bigram_lp(e.domain), "dom_consonant_run": consonant_run(e.domain),
        "dom_vowel_ratio": sum(c in "aeiou" for c in e.domain) / max(1, len(e.domain)),
        "dom_digits": sum(c.isdigit() for c in e.domain), "dom_hyphens": e.domain.count("-"),
        "host_word_cover": word_cover(host.replace(".", "-")),
        "rest_word_cover": word_cover(rest.lower()),
        "rest_bigram_lp": bigram_lp(rest),
        "tld": e.suffix or "?", "registered_domain": reg or host,
        # lo schema http/https NON entra nel modello: nelle fonti disponibili dice da quale
        # lista viene l'URL, non se e' pericoloso. Resta come "_is_https" per i report.
        "url_text": host + rest,
        "host_text": host, "path_text": rest,
    }
    return f

NUMERIC = None
def featurize(urls):
    df = pd.DataFrame(extract(u) for u in urls)
    global NUMERIC
    NUMERIC = [c for c in df.columns if c not in ("tld", "registered_domain", "url_text", "host_text", "path_text")
               and not c.startswith("_")]
    return df

if __name__ == "__main__":
    for u in ["https://www.poste.it/",
              "http://paypal.com.secure-login.verify-account.tk/webscr?cmd=_login",
              "https://github.com/scikit-learn/scikit-learn/pull/28123",
              "http://192.168.4.11:8080/wp-content/dhl/tracking.php"]:
        f = extract(u)
        print(f"{u[:58]:60s} brand_mismatch={f['brand_mismatch']} sub={f['n_subdomains']} sens={f['n_sensitive_words']} ent={f['url_entropy']:.2f}")
