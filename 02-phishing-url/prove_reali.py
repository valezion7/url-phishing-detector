"""Prova di accettazione su URL veri e conosciuti, italiani e non.

Non e' una metrica: e' il controllo che ha trovato i due errori peggiori del progetto
(la pagina di login di un cliente e la home di Intesa Sanpaolo date per phishing).
Le metriche non li avrebbero mai visti, perche' nei dati quelle categorie mancavano.

  python prove_reali.py
"""
import pathlib, sys
import check_url as C

# Lista pubblica. Se hai i tuoi domini da controllare, mettili in `prove_mie.txt`
# (un URL per riga): il file e' ignorato da git e viene aggiunto in coda ai BUONI.
BUONI = [
    "https://accounts.spotify.com/it/login",
    "https://www.zalando.it/moda-uomo/",
    "https://www.intesasanpaolo.com/it/persone-e-famiglie.html",
    "https://www.poste.it/",
    "https://www.agenziaentrate.gov.it/portale/rimborso-2026?id=99",
    "https://github.com/scikit-learn/scikit-learn/pull/28123",
    "https://accounts.google.com/signin/v2/identifier?service=mail",
    "https://www.amazon.it/gp/css/order-history?ref_=nav_orders_first",
    "https://un-scrabbled.com",
    "https://www.subito.it/annunci-italia/vendita/usato/",
    "https://www.ilpost.it/2026/09/22/",
]
CATTIVI_VERI = [        # presi dai feed pubblici, quindi realmente segnalati
    "http://103.125.106.2/intesasanpaolo.com/privati/okeylogin.html?securessl=true",
    "https://www.wbeuvvfx.com/",
    "http://appsx.duckdns.org/~gestorvt/xuione",
]
CATTIVI_COSTRUITI = [   # scritti a mano sui motivi classici: servono a vedere se il
    "http://paypal.com.secure-login.verify-account.tk/webscr?cmd=_login",   # modello
    "https://intesasanpaolo.it-sicurezza-clienti.top/accesso/verifica.php?id=8812",  # riconosce
    "https://poste-it.secure-login.xyz/it/accedi",                          # il meccanismo,
    "http://192.168.4.11:8080/wp-content/dhl/tracking.php",                 # non a misurare
    "https://amaz0n-support.verify-account.cc/login",                       # niente
]
CATTIVI = CATTIVI_VERI + CATTIVI_COSTRUITI

_mie = pathlib.Path(__file__).parent / "prove_mie.txt"
if _mie.exists():
    BUONI += [l.strip() for l in _mie.read_text(encoding="utf-8").splitlines() if l.strip()]

def main():
    urls = BUONI + CATTIVI
    _, p = C.score(urls)
    errori = 0
    print(f"soglia sospetto {C.TH['threshold_fpr1']:.3f} | soglia phishing {C.TH['threshold_fpr01']:.3f}\n")
    for etichetta, gruppo, atteso in (("BUONI", BUONI, 0), ("CATTIVI", CATTIVI, 1)):
        print(f"--- {etichetta}")
        for u in gruppo:
            v = p[urls.index(u)]
            g = C.giudizio(v)
            sbagliato = (atteso == 0 and g == "PHISHING") or (atteso == 1 and g == "ok")
            errori += sbagliato
            print(f"  {'X' if sbagliato else ' '} {v:.3f}  {g:9s}  {u[:76]}")
        print()
    print(f"errori gravi: {errori}/{len(urls)}")
    return errori

if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
