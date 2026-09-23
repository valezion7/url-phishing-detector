# Rilevatore di URL di phishing

*Documentazione in italiano. English documentation: [repository README](../README.md).*


Dice se un link e' phishing **guardando solo la stringa dell'URL**: nessuna richiesta di
rete, nessun fetch della pagina, nessun servizio esterno. Si puo' chiamare prima che
l'utente clicchi, dentro un form, in un filtro di posta o in un pannello CMS.

Gira sulla CPU con l'Anaconda gia' installato. Addestramento completo: pochi minuti.
Giudizio su un URL: microsecondi.

```bash
P="C:/ProgramData/anaconda3/python.exe"
$P check_url.py "https://paypal.com.secure-login.verify.tk/webscr?cmd=_login"
$P check_url.py --file lista_di_link.txt
```

## Da dove vengono i dati

Niente dataset preconfezionato: le liste pubbliche piu' usate hanno difetti che gonfiano
i risultati (vedi `DIARIO.md`, voce 0). Tutto raccolto dal vivo con `collect_data.py`.

| classe | fonte | cosa e' |
|---|---|---|
| phishing | Phishing.Database (lista ACTIVE) | URL di phishing attivi, aggiornati ogni giorno |
| benigni | Wikipedia `exturlusage` | link esterni realmente citati dalle voci: percorsi veri, domini diversissimi |
| benigni | Wikipedia per TLD | stessa fonte cercata TLD per TLD, per coprire `.io .ai .dev .xyz .app ...` |
| benigni | sitemap di siti Tranco | pagine profonde di siti veri, anche su TLD moderni |
| benigni | **pagine di accesso vere** | 3.941 pagine `/login`, `/account`, `/checkout`... verificate con HTTP 200 |
| benigni | Wikipedia it/de/fr/es | perche' il web buono non e' tutto in inglese |
| benigni | homepage Tranco | campione lungo tutta la classifica, non solo i famosi |

Tenuti **fuori dal training**, come prova indipendente:
OpenPhish (phishing verificato del giorno), Hacker News e lobste.rs (link postati da
persone lo stesso giorno), URLhaus (URL di malware: minaccia diversa, serve a vedere se
il modello generalizza).

Regole di composizione, tutte nate da un errore vero:
- massimo 3 URL per dominio registrato (le liste di phishing ripetono lo stesso host);
- ogni TLD usato dal phishing deve avere anche dei benigni;
- benigni di forme diverse (homepage, pagina profonda, pagina di accesso);
- le pagine di accesso vere entrano tutte e **ripetute fino al 7%** dei benigni: sono
  poche ma sono la categoria che il phishing imita, e alla loro frequenza naturale
  (1%) sparivano nel campione;
- schema `http/https` escluso dal modello e slash finale e `www.` normalizzati: dicono da
  quale lista viene l'URL, non se e' pericoloso.

## Come giudica

Due modelli in parallelo, poi la media delle probabilita':

1. **regressione logistica sui caratteri** — tre viste TF-IDF: n-grammi del nome host,
   n-grammi del percorso, parole intere;
2. **gradient boosting su 60 feature calcolate** — lunghezze, sottodomini, cifre, entropia,
   marchi fuori posto, host condivisi, e due misure di *leggibilita'* del nome a dominio
   (quanta parte e' fatta di parole vere, quanto i bigrammi somigliano a una lingua).
   Sono quelle che distinguono `un-scrabbled.com` da `wbeuvvfx.com`.

Confrontati nel run: regressione logistica, albero decisionale, random forest,
boosting, il modello a n-grammi, l'ensemble a voto e uno stacking con meta-modello.

## Come e' valutato

Tre prove, sempre nell'ordine dalla piu' facile alla piu' onesta:

1. **split casuale** — il numero che si legge di solito;
2. **split per dominio** — nessun dominio compare sia in addestramento che in prova;
3. **holdout indipendente** — phishing verificato di oggi contro link postati oggi da
   persone, piu' gli URL di malware di URLhaus.

Metrica di riferimento: **quanto phishing prendo, a parita' di link buoni bloccati**.
Un filtro che blocca link deve sbagliare pochissimo sui buoni, quindi l'accuratezza non
serve a niente: conta il recall a FPR dell'1%.

## Risultati

Dataset finale: **260.000 URL** (metà phishing attivo, metà web buono), 195.000 domini circa.

### Test interno, split per dominio (nessun dominio in comune fra addestramento e prova)

| modello | AUC split casuale | AUC split per dominio | PR-AUC | phishing preso bloccando l'1% | ... lo 0,1% |
|---|---|---|---|---|---|
| Logistic Regression (feature) | 0,925 | 0,924 | 0,928 | 43,9% | 21,4% |
| Decision Tree (feature) | 0,865 | 0,865 | 0,867 | 33,9% | 13,1% |
| Random Forest (feature) | 0,962 | 0,953 | 0,957 | 58,8% | 32,6% |
| HistGradientBoosting (feature) | 0,977 | 0,974 | 0,976 | 72,4% | 50,2% |
| Logistic Regression (n-grammi di caratteri) | 0,987 | 0,984 | 0,986 | 82,7% | 59,9% |
| Ensemble a voto | 0,989 | 0,987 | 0,988 | 84,2% | 68,4% |
| **Stacking (è quello spedito)** | **0,989** | **0,987** | **0,989** | **84,4%** | **69,8%** |

### Holdout indipendente — la prova che conta

2.210 link postati oggi da persone (Hacker News, lobste.rs) contro 193 URL di phishing
verificati oggi (OpenPhish), nessun dominio in comune con l'addestramento.

**AUC 0,948.**

| link buoni bloccati | phishing riconosciuto | soglia | malware URLhaus riconosciuto |
|---|---|---|---|
| 0,50% | 45,6% | 0,984 | 91,1% |
| **0,90%** | **53,4%** | 0,977 | 92,8% |
| 1,99% | 62,7% | 0,958 | 94,3% |
| 4,89% | 78,2% | 0,844 | 96,3% |
| 9,86% | 86,0% | 0,542 | 97,3% |

A soglia 0,50: prende 166 phishing su 193 (86,0%) e blocca 234 link buoni su 2.210 (10,6%).

Con 193 phishing verificati, l'intervallo al 95% sul 53,4% va da 46,6% a 60,6%: è una stima
onesta, non una misura di precisione. Tarando la soglia su metà dell'holdout e misurandola
sull'altra metà: **0,58% di buoni bloccati, 50,0% di phishing preso**.

L'ultima colonna è una prova di trasferimento: 13.427 URL di **malware** (URLhaus), una
minaccia che il modello non ha mai visto in addestramento, riconosciuta al 93%.

### Prova di accettazione su URL veri

`prove_reali.py`: 21 URL conosciuti (domini di clienti, Intesa Sanpaolo, Poste, Agenzia
Entrate, GitHub, Amazon, Subito, più phishing dai feed e costruito). **3 errori su 21**,
tutti noti: `accounts.google.com/signin` resta un falso positivo, e due phishing su dominio
anonimo restano appena sotto la soglia conservativa.

## Limiti dichiarati

- **Domini legittimi compromessi**: se il phishing sta su un sito vero bucato, la stringa
  spesso non contiene nessun segnale. E' il limite del problema, non del modello.
- Le due misure di leggibilita' usano un vocabolario **inglese**: un dominio in una lingua
  senza parole inglesi parte svantaggiato.
- Il modello non conosce **eta' del dominio, certificato, reputazione IP**: in produzione
  si affiancano, e alzano molto il risultato.
- I feed di phishing usati sono in prevalenza di lingua inglese e portoghese/brasiliana;
  su campagne italiane va riaddestrato con dati italiani.
- La lista dei marchi e dei domini noti viene da Tranco: va aggiornata, altrimenti i
  marchi nuovi non vengono riconosciuti.

## File

```
collect_data.py    scarica le fonti pubbliche (cache in data/raw/)
build_dataset.py   compone il dataset togliendo le scorciatoie  [--naturale]
build_holdout.py   rifa' solo l'holdout indipendente
features.py        le feature dalla sola stringa (usate anche in produzione)
test_features.py   controllo minimo: cade se l'estrazione si rompe
prove_reali.py     prova di accettazione su URL veri e conosciuti
train_phishing.py  7 modelli x 2 split, metriche e figure       [--naturale] [--prova N]
eval_holdout.py    la prova su dati di altri
riepilogo.py       tutti i numeri finali in un posto solo
componi_pagina.py  genera la pagina pubblica dai numeri veri
check_url.py       la riga di comando da usare davvero
DIARIO.md          gli otto inganni trovati durante il lavoro + due correzioni di troppo

data/raw/          le liste scaricate (cache: riesegui collect_data.py con --force)
outputs_naturale/  il modello spedito, le metriche e le figure
outputs/           l'esperimento del dataset "pareggiato", tenuto per il confronto
```
