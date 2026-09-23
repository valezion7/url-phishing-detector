"""Controllo minimo: se una di queste cade, l'estrazione delle feature si e' rotta.
    python test_features.py
"""
import features as F


def test_canonicalizzazione():
    """Slash finale e www non devono cambiare niente: sono la scorciatoia n.1 del DIARIO."""
    a = F.extract("https://www.esempio.com/")
    b = F.extract("http://esempio.com")
    assert a["url_text"] == b["url_text"] == "esempio.com", (a["url_text"], b["url_text"])
    assert a["path_len"] == 0 and a["n_subdomains"] == 0


def test_schema_fuori_dal_modello():
    """Lo schema non deve finire fra le feature del modello."""
    df = F.featurize(["https://a.com/x", "http://a.com/x"])
    assert "_is_https" not in F.NUMERIC and "is_https" not in F.NUMERIC
    assert df.loc[0, "url_text"] == df.loc[1, "url_text"]


def test_marchi():
    assert F.extract("https://paypal.com.login-verify.tk/x")["brand_mismatch"] == 1
    assert F.extract("https://www.paypal.com/it/signin")["brand_mismatch"] == 0
    assert F.extract("https://roblox.com.do/games/1")["brand_foreign_tld"] == 1
    assert F.extract("https://roblox.com/games/1")["brand_foreign_tld"] == 0
    assert "code" not in F.brands() and "paypal" in F.brands()   # niente parole comuni fra i marchi


def test_leggibilita():
    """Distingue un sito piccolo ma dal nome parlante da un dominio di lettere a caso."""
    for buono in ("proudsend", "3dassetstudio", "un-scrabbled"):
        assert F.word_cover(buono) >= .8, buono
    for cattivo in ("wbeuvvfx", "wydhnxbh"):
        assert F.word_cover(cattivo) == 0 and F.bigram_lp(cattivo) < -6, cattivo


def test_non_esplode():
    for u in ("", "   ", "http://", "javascript:alert(1)", "https://[::1]:8080/x",
              "https://a:99999999/x", "https://" + "a" * 2000 + ".com"):
        F.extract(u)


if __name__ == "__main__":
    for nome, fn in sorted(globals().items()):
        if nome.startswith("test_"):
            fn(); print("ok", nome)
    print("tutto a posto")
