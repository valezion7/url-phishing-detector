"""Raccoglie URL veri e attuali da fonti pubbliche. Ogni fonte finisce in data/raw/,
riesegui quando vuoi: quello che c'e' gia' non viene riscaricato (--force per rifarlo).

PHISHING  Phishing.Database (lista ACTIVE), OpenPhish (feed community), URLhaus (abuse.ch)
BENIGNI   Wikipedia exturlusage, sitemap di domini Tranco, link postati su Hacker News

Le tre fonti benigne servono a coprire forme diverse (homepage, pagina profonda,
articolo, pagina commerciale): se i benigni fossero solo homepage il modello
imparerebbe "ha un path => phishing" invece di imparare il phishing.
"""
import concurrent.futures as cf, gzip, io, json, pathlib, re, sys, time, zipfile
import requests

# priorita' bassa: la macchina sta facendo anche altro (Blender)
try:
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception: pass

RAW = pathlib.Path(__file__).parent / "data/raw"; RAW.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "url-classifier-research/1.0 (contatto: valerio.bonetti07@gmail.com)"}
FORCE = "--force" in sys.argv
S = requests.Session(); S.headers.update(UA)
# Un solo contesto TLS per tutte le connessioni. Senza questo, ogni host nuovo fa
# ricaricare e riparsare il bundle dei certificati: su migliaia di domini diventa il
# collo di bottiglia e mangia decine di core per un lavoro che e' solo di rete.
import ssl, certifi
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
_CTX = ssl.create_default_context(cafile=certifi.where())

class _Adattatore(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **kw):
        self.poolmanager = PoolManager(num_pools=connections, maxsize=maxsize, block=block,
                                       ssl_context=_CTX, **kw)

_ad = _Adattatore(pool_connections=64, pool_maxsize=64, max_retries=0)
S.mount("https://", _ad); S.mount("http://", _ad)

def cached(name):
    def deco(fn):
        def wrap(*a, **k):
            f = RAW / name
            if f.exists() and not FORCE:
                n = sum(1 for _ in f.open(encoding="utf-8", errors="ignore"))
                print(f"= {name:22s} {n:>7,} righe (cache)"); return f
            t = time.time(); fn(f, *a, **k)
            n = sum(1 for _ in f.open(encoding="utf-8", errors="ignore"))
            print(f"+ {name:22s} {n:>7,} righe ({time.time()-t:.0f}s)"); return f
        return wrap
    return deco

# ---------------- phishing ----------------
@cached("phishing_db.txt")
def phishing_db(f):
    r = S.get("https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt", timeout=180)
    r.raise_for_status(); f.write_text(r.text, encoding="utf-8")

@cached("openphish.txt")
def openphish(f):
    r = S.get("https://openphish.com/feed.txt", timeout=60); r.raise_for_status()
    f.write_text(r.text, encoding="utf-8")

@cached("urlhaus.txt")
def urlhaus(f):
    r = S.get("https://urlhaus.abuse.ch/downloads/csv_recent/", timeout=120); r.raise_for_status()
    urls = [l.split('","')[2] for l in r.text.splitlines() if l.startswith('"') and len(l.split('","')) > 3]
    f.write_text("\n".join(urls), encoding="utf-8")

# ---------------- benigni ----------------
@cached("tranco.txt")
def tranco(f, n=200_000):
    r = S.get("https://tranco-list.eu/top-1m.csv.zip", timeout=300); r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        rows = z.read(z.namelist()[0]).decode().splitlines()[:n]
    f.write_text("\n".join(l.split(",")[1] for l in rows if "," in l), encoding="utf-8")

@cached("wikipedia.txt")
def wikipedia(f, pages=160):
    """exturlusage: link esterni realmente citati dalle voci. Path veri, domini diversissimi."""
    out, cont = [], None
    for proto in ("https", "http"):
        cont = None
        for _ in range(pages if proto == "https" else pages // 3):
            p = {"action": "query", "list": "exturlusage", "eulimit": "500", "format": "json",
                 "euprotocol": proto, "eunamespace": "0"}
            if cont: p["eucontinue"] = cont
            try:
                j = S.get("https://en.wikipedia.org/w/api.php", params=p, timeout=45).json()
            except Exception:
                break
            out += [e["url"] for e in j.get("query", {}).get("exturlusage", [])]
            cont = j.get("continue", {}).get("eucontinue")
            if not cont: break
            time.sleep(.12)
    f.write_text("\n".join(out), encoding="utf-8")

@cached("sitemaps.txt")
def sitemaps(f, n_domains=4000, per_domain=25):
    """Pagine profonde di siti veri e popolari: login, prodotti, articoli, account.
    Sono proprio le forme che il phishing imita, senza di loro il modello barerebbe."""
    doms = (RAW / "tranco.txt").read_text(encoding="utf-8").splitlines()[:n_domains]
    return _sitemap_crawl(f, doms, per_domain)

def _sitemap_crawl(f, doms, per_domain, workers=16):
    sm_re = re.compile(r"(?im)^\s*sitemap:\s*(\S+)")
    loc_re = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.I)

    def get(u, limit=3_000_000):
        r = S.get(u, timeout=6, stream=True, allow_redirects=True)
        if r.status_code != 200: return None
        b = r.raw.read(limit, decode_content=True)
        if u.endswith(".gz") or b[:2] == b"\x1f\x8b":
            try: b = gzip.decompress(b)
            except Exception: return None
        return b.decode("utf-8", "ignore")

    def one(d):
        try:
            txt = get(f"https://{d}/robots.txt", 200_000)
            sms = sm_re.findall(txt)[:2] if txt else []
            if not sms: sms = [f"https://{d}/sitemap.xml"]
            got = []
            for sm in sms:
                body = get(sm)
                if not body: continue
                locs = loc_re.findall(body)
                nested = [l for l in locs if ".xml" in l.lower()]
                if nested and len(nested) > len(locs) * .8:      # indice di sitemap
                    body = get(nested[0]) or ""
                    locs = loc_re.findall(body)
                got += [l for l in locs if ".xml" not in l.lower()]
                if len(got) >= per_domain: break
            return got[:per_domain]
        except Exception:
            return []

    out = []
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, got in enumerate(ex.map(one, doms), 1):
            out += got
            if i % 250 == 0: print(f"   sitemap {i}/{len(doms)} domini -> {len(out):,} url", flush=True)
    f.write_text("\n".join(out), encoding="utf-8")

@cached("wikipedia_tld.txt")
def wikipedia_tld(f, pages=14):
    """Link esterni di Wikipedia cercati per TLD. Serve a coprire .io .ai .dev .xyz ...:
    nei primi dati i benigni non ne avevano nemmeno uno e il modello bloccava
    qualsiasi sito su un TLD moderno."""
    tlds = ("io ai dev app xyz online site top live shop club space cloud tech me co sh social "
            "link fun store blog page pro info biz cc tv ws su icu vip work world today life "
            "news media digital agency studio design art fyi gg one run tools").split()
    out = []
    for tld in tlds:
        for proto in ("https", "http"):
            cont, empty = None, 0
            for _ in range(pages):
                q = {"action": "query", "list": "exturlusage", "eulimit": "500", "format": "json",
                     "euquery": f"*.{tld}", "euprotocol": proto, "eunamespace": "0"}
                if cont: q["eucontinue"] = cont
                try: j = S.get("https://en.wikipedia.org/w/api.php", params=q, timeout=45).json()
                except Exception: break
                got = [e["url"] for e in j.get("query", {}).get("exturlusage", [])]
                out += got
                empty = empty + 1 if not got else 0
                cont = j.get("continue", {}).get("eucontinue")
                if not cont or empty >= 4: break
                time.sleep(.1)
        print(f"   {tld:8s} totale {len(out):,}", flush=True)
    f.write_text(chr(10).join(out), encoding="utf-8")

@cached("wikipedia_lingue.txt")
def wikipedia_lingue(f, pagine=70):
    """Link esterni dalle Wikipedia in italiano, tedesco, francese e spagnolo.

    Senza questi il modello e' addestrato su un web quasi tutto inglese e bolla come
    sospette le pagine in altre lingue: la pagina vera di Intesa Sanpaolo prendeva 0,94.
    """
    out = []
    for lingua in ("it", "de", "fr", "es"):
        prima = len(out)
        for proto in ("https", "http"):
            cont = None
            for _ in range(pagine if proto == "https" else pagine // 3):
                q = {"action": "query", "list": "exturlusage", "eulimit": "500", "format": "json",
                     "euprotocol": proto, "eunamespace": "0"}
                if cont: q["eucontinue"] = cont
                try: j = S.get(f"https://{lingua}.wikipedia.org/w/api.php", params=q, timeout=45).json()
                except Exception: break
                out += [e["url"] for e in j.get("query", {}).get("exturlusage", [])]
                cont = j.get("continue", {}).get("eucontinue")
                if not cont: break
                time.sleep(.1)
        print(f"   {lingua}: +{len(out)-prima:,} (totale {len(out):,})", flush=True)
    f.write_text(chr(10).join(dict.fromkeys(out)), encoding="utf-8")

@cached("sitemaps_modern.txt")
def sitemaps_modern(f, per_tld=400, per_domain=6):
    """Secondo giro di sitemap, ma sui TLD moderni (.io .ai .dev .xyz .app .online ...).
    Senza questi fra i benigni non c'e' un solo .ai e il modello blocca mezza Hacker News."""
    import tldextract
    ext = tldextract.TLDExtract(cache_dir=".tld_cache")
    want = {"io","ai","dev","app","xyz","online","site","top","live","shop","club","space",
            "cloud","tech","me","co","sh","social","link","fun","store","blog","page","pro"}
    doms, seen = [], {}
    for d in (RAW / "tranco.txt").read_text(encoding="utf-8").splitlines():
        t = ext(d).suffix
        if t in want and seen.get(t, 0) < per_tld:
            seen[t] = seen.get(t, 0) + 1; doms.append(d)
    print("   domini moderni da provare:", len(doms), seen)
    globals()["_SM_DOMS"] = doms
    _sitemap_crawl(f, doms, per_domain, workers=48)

@cached("hackernews.txt")
def hackernews(f):
    ids = []
    for lst in ("topstories", "newstories", "beststories"):
        ids += S.get(f"https://hacker-news.firebaseio.com/v0/{lst}.json", timeout=30).json()
    ids = list(dict.fromkeys(ids))
    def one(i):
        try: return (S.get(f"https://hacker-news.firebaseio.com/v0/item/{i}.json", timeout=15).json() or {}).get("url")
        except Exception: return None
    with cf.ThreadPoolExecutor(20) as ex:
        urls = [u for u in ex.map(one, ids) if u]
    f.write_text("\n".join(urls), encoding="utf-8")

@cached("pagine_accesso.txt")
def pagine_accesso(f, n_domini=6_000, per_dominio=3):
    """Pagine di accesso VERE di siti veri, verificate una per una (HTTP 200).

    Serve perche' il phishing imita proprio queste: /login, /account, /verify.
    Senza di loro il modello impara "percorso di login = phishing" e bolla come
    truffa la pagina di accesso di qualunque azienda -- successo davvero, con un
    dominio reale di un cliente. Teniamo l'URL FINALE dopo i redirect, che e'
    spesso ancora piu' simile a un phishing (accounts.x.com/signin?next=...).

    Le prove sono appiattite in una coda unica (dominio, percorso): farle in
    sequenza per dominio lasciava i thread fermi ad aspettare.
    """
    import random
    random.seed(7)
    PERCORSI = ["/login", "/signin", "/sign-in", "/account", "/account/login", "/user/login",
                "/my-account", "/customer/account/login", "/wp-login.php", "/register",
                "/signup", "/sign-up", "/password/reset", "/forgot-password", "/checkout",
                "/cart", "/profile", "/settings", "/dashboard", "/admin/login", "/auth/login",
                "/secure/login", "/verify", "/subscribe", "/billing", "/orders"]
    doms = (RAW / "tranco.txt").read_text(encoding="utf-8").splitlines()
    scelti = [doms[i] for i in sorted({int(random.uniform(0, 1) ** 2 * min(400_000, len(doms)))
                                       for _ in range(n_domini * 2)})][:n_domini]
    prove = [(d, perc) for d in scelti for perc in random.sample(PERCORSI, per_dominio)]
    random.shuffle(prove)          # cosi' non martelliamo lo stesso host di fila

    def una(dp):
        d, perc = dp
        try:
            r = S.get(f"https://{d}{perc}", timeout=5, allow_redirects=True, stream=True)
            r.close()
            if r.status_code == 200 and len(r.url) > len(f"https://{d}/") + 2:
                return r.url
        except Exception:
            pass
        return None

    out = []
    with cf.ThreadPoolExecutor(64) as ex:
        for k, u in enumerate(ex.map(una, prove), 1):
            if u: out.append(u)
            if k % 3000 == 0: print(f"   {k}/{len(prove)} prove -> {len(out):,} pagine", flush=True)
    f.write_text(chr(10).join(dict.fromkeys(out)), encoding="utf-8")

@cached("pagine_accesso2.txt")
def pagine_accesso2(f, n_domini=9_000, per_dominio=3):
    """Secondo giro di pagine di accesso, su domini piu' popolari (rango 1-150.000):
    hanno quasi sempre un'area riservata, quindi il raccolto e' piu' ricco."""
    import random
    random.seed(11)
    g = globals()
    PERCORSI = ["/login", "/signin", "/sign-in", "/account", "/account/login", "/user/login",
                "/my-account", "/customer/account/login", "/wp-login.php", "/register",
                "/signup", "/password/reset", "/forgot-password", "/checkout", "/cart",
                "/profile", "/settings", "/dashboard", "/auth/login", "/secure/login",
                "/verify", "/subscribe", "/billing", "/orders", "/members", "/portal"]
    doms = (RAW / "tranco.txt").read_text(encoding="utf-8").splitlines()[:150_000]
    scelti = random.sample(doms, min(n_domini, len(doms)))
    prove = [(d, perc) for d in scelti for perc in random.sample(PERCORSI, per_dominio)]
    random.shuffle(prove)

    def una(dp):
        d, perc = dp
        try:
            r = S.get(f"https://{d}{perc}", timeout=5, allow_redirects=True, stream=True); r.close()
            if r.status_code == 200 and len(r.url) > len(f"https://{d}/") + 2:
                return r.url
        except Exception:
            pass
        return None

    out = []
    with cf.ThreadPoolExecutor(64) as ex:
        for k, u in enumerate(ex.map(una, prove), 1):
            if u: out.append(u)
            if k % 4000 == 0: print(f"   {k}/{len(prove)} prove -> {len(out):,} pagine", flush=True)
    f.write_text(chr(10).join(dict.fromkeys(out)), encoding="utf-8")

@cached("hackernews_recenti.txt")
def hackernews_recenti(f, quanti=25_000):
    """Scorre gli ultimi N elementi di Hacker News e tiene i link: benigni freschi,
    scritti da persone, su domini che il training non ha mai visto."""
    top = S.get("https://hacker-news.firebaseio.com/v0/maxitem.json", timeout=30).json()
    def one(i):
        try: 
            it = S.get(f"https://hacker-news.firebaseio.com/v0/item/{i}.json", timeout=12).json() or {}
            return it.get("url")
        except Exception: return None
    out = []
    with cf.ThreadPoolExecutor(32) as ex:
        for k, u in enumerate(ex.map(one, range(top - quanti, top)), 1):
            if u: out.append(u)
            if k % 5000 == 0: print(f"   {k}/{quanti} -> {len(out)} link", flush=True)
    f.write_text(chr(10).join(dict.fromkeys(out)), encoding="utf-8")

@cached("hackernews_storico.txt")
def hackernews_storico(f, da=65_000, a=25_000):
    """Altri link freschi di Hacker News, piu' indietro nel tempo: serve solo a rendere
    meno rumorosa la stima dei falsi positivi sull'holdout."""
    top = S.get("https://hacker-news.firebaseio.com/v0/maxitem.json", timeout=30).json()
    def one(i):
        try: return (S.get(f"https://hacker-news.firebaseio.com/v0/item/{i}.json", timeout=12).json() or {}).get("url")
        except Exception: return None
    out = []
    with cf.ThreadPoolExecutor(32) as ex:
        for k, u in enumerate(ex.map(one, range(top - da, top - a)), 1):
            if u: out.append(u)
            if k % 10000 == 0: print(f"   {k}/{da-a} -> {len(out)} link", flush=True)
    f.write_text(chr(10).join(dict.fromkeys(out)), encoding="utf-8")

@cached("lobsters.txt")
def lobsters(f):
    """Secondo campione di link postati da persone, indipendente da Hacker News."""
    out = []
    for page in range(1, 9):
        for lst in ("hottest", "newest"):
            try: j = S.get(f"https://lobste.rs/{lst}.json?page={page}", timeout=30).json()
            except Exception: continue
            out += [x["url"] for x in j if x.get("url")]
            time.sleep(.3)
    f.write_text(chr(10).join(dict.fromkeys(out)), encoding="utf-8")

if __name__ == "__main__":
    phishing_db(); openphish(); urlhaus(); tranco(); wikipedia(); hackernews(); sitemaps()
    print("\nfatto ->", RAW)
