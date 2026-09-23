# Otto modi in cui un classificatore mi ha mentito

*[Read in English](README.md)*

**[Provalo dal vivo](https://phishing.studiobeezy.com)** · **[Modello su Hugging Face](https://huggingface.co/valezion/url-phishing-detector)**

Un rilevatore di URL di phishing che giudica un link **dalla sola stringa** — nessuna chiamata
di rete, nessun fetch della pagina — più il registro di ogni scorciatoia che si è preso mentre
lo costruivo, e di come ognuna è stata scoperta.

Il punto di questo repository non è il modello. È che il test del modello diceva **0,99**
mentre bloccava un quarto del web buono, e che l'unica cosa che se ne è accorta è stato un
insieme di dati raccolto da qualcun altro.

| | ROC-AUC |
|---|---|
| split casuale | 0,989 |
| **split per dominio** (nessun dominio in comune fra addestramento e prova) | **0,987** |
| **holdout indipendente** (phishing verificato di oggi contro link postati oggi) | **0,948** |

Sull'holdout indipendente, bloccando lo **0,9% dei link buoni** riconosce il **53% del
phishing fresco** (intervallo 95%: 46,6–60,6%) e il **93% degli URL di malware di URLhaus**,
che è una minaccia diversa, mai vista in addestramento. Accettando il 4,9% di falsi positivi
arriva al 78%.

Tutto gira su CPU con uno stack scientifico Python standard. Addestramento completo: circa 17
minuti. Giudizio su un URL: microsecondi.

---

## La parte che vale la pena leggere

**[`02-phishing-url/DIARIO.md`](02-phishing-url/DIARIO.md)** — otto volte in cui il modello ha
imparato qualcosa che non c'entrava niente con il phishing, e due "miglioramenti" che hanno
alzato il punteggio interno peggiorando quello vero.

Un assaggio:

- Ha imparato che **un URL che finisce con `/` è buono**, perché le homepage benigne le avevo
  generate io e avevano tutte lo slash finale. I link veri postati dalle persone no.
- Ha imparato che **`http://` vuol dire buono**, perché i miei URL benigni venivano da
  citazioni di Wikipedia, piene di vecchi link `http`. L'esatto contrario della realtà.
- Fra i benigni non c'era **nemmeno un `.ai`, `.io` o `.dev`**, mentre su Hacker News sono il
  15% dei link. Ogni sito su un TLD moderno veniva bloccato.
- Ha imparato che **un dominio che non conosce è phishing**, per un `head()` al posto di un
  `sample()`: le homepage benigne erano tutte fra i primi 15.000 siti del mondo.
- Ha imparato che **un percorso di login è phishing**, perché fra i dati benigni le pagine di
  accesso quasi non esistevano. Il pannello reale di un cliente prendeva il 99,5%.
- Separare le feature testuali in tre viste (host / percorso / parole) alzava l'AUC interna da
  0,9841 a 0,9849 e **faceva crollare quella esterna da 0,950 a 0,907**.

Nessuno di questi problemi è stato segnalato dal test interno. Nemmeno uno.

## Che cosa guarda davvero il modello

Due modelli uniti da un meta-modello (stacking):

1. **Regressione logistica su n-grammi di caratteri** (3–5, `char_wb`) di tutto l'URL.
2. **Gradient boosting su 62 feature calcolate** — lunghezze, numero di sottodomini, cifre,
   entropia, un marchio noto che compare nel nome host sotto un dominio che non è il suo, host
   condivisi, e due misure di quanto il nome a dominio **si legge**.

Quelle due ultime sono ciò che distingue un sito piccolo da un dominio usa e getta, cosa che
la lunghezza da sola non può fare:

| dominio | copertura in parole vere | plausibilità dei bigrammi |
|---|---|---|
| `proudsend` | 1,00 | −3,6 |
| `3dassetstudio` | 0,85 | −4,2 |
| `un-scrabbled` | 0,82 | −3,5 |
| `webufexkp` | 0,33 | −6,0 |
| `wbeuvvfx` | 0,00 | −8,0 |

**Esclusi apposta** dal modello: lo schema `http`/`https`, lo slash finale e il prefisso
`www.`. Dicono da quale lista viene un URL, non se è pericoloso.

## Per iniziare

```bash
pip install scikit-learn pandas numpy matplotlib joblib tldextract requests

cd 02-phishing-url
python collect_data.py             # ricostruisce il dataset dai feed pubblici (ore, quasi tutte di attesa)
python build_dataset.py --naturale
python build_holdout.py
python train_phishing.py --naturale
python riepilogo.py --naturale

python check_url.py "https://paypal.com.secure-login.verify.tk/webscr?cmd=_login"
#   punteggio 0.994  ->  PHISHING
#    - un marchio noto compare fuori dal dominio registrato
#    - 3 sottodomini incatenati
#    - parole da pagina di accesso (login/verify/secure...)
```

`python check_url.py --file lista.txt` classifica una lista intera.

## Risultati per esteso

### Test interno, split per dominio

Il 20% dei *domini* è tenuto fuori per intero, quindi il modello non può riconoscere un host
che ha già visto.

| modello | AUC split casuale | AUC per dominio | PR-AUC | preso @FPR 1% | @FPR 0,1% |
|---|---|---|---|---|---|
| Regressione logistica (feature) | 0,925 | 0,924 | 0,928 | 43,9% | 21,4% |
| Albero decisionale (feature) | 0,865 | 0,865 | 0,867 | 33,9% | 13,1% |
| Random forest (feature) | 0,962 | 0,953 | 0,957 | 58,8% | 32,6% |
| HistGradientBoosting (feature) | 0,977 | 0,974 | 0,976 | 72,4% | 50,2% |
| Regressione logistica (n-grammi) | 0,987 | 0,984 | 0,986 | 82,7% | 59,9% |
| Ensemble a voto | 0,989 | 0,987 | 0,988 | 84,2% | 68,4% |
| **Stacking (è quello spedito)** | **0,989** | **0,987** | **0,989** | **84,4%** | **69,8%** |

### Holdout indipendente — quello che conta

2.210 link postati lo stesso giorno da persone (Hacker News, lobste.rs) contro 193 URL di
phishing verificati lo stesso giorno (OpenPhish). Nessun dominio in comune con
l'addestramento.

| link buoni bloccati | phishing fresco riconosciuto | soglia | malware URLhaus riconosciuto |
|---|---|---|---|
| 0,50% | 45,6% | 0,984 | 91,1% |
| **0,90%** | **53,4%** | 0,977 | 92,8% |
| 1,99% | 62,7% | 0,958 | 94,3% |
| 4,89% | 78,2% | 0,844 | 96,3% |
| 9,86% | 86,0% | 0,542 | 97,3% |

Con soli 193 URL di phishing verificati, il 53,4% porta un intervallo al 95% da 46,6% a 60,6%.
È una stima onesta, non una misura di precisione, e il repository lo dice ovunque.

Tarando la soglia su metà dell'holdout e misurandola sull'altra metà:
**0,58% di link buoni bloccati, 50,0% di phishing preso.**

## Da dove vengono i dati

Nessun dataset preconfezionato. Il dataset accademico più citato per questo compito (PhiUSIIL,
235.795 URL) è inutilizzabile: tutti i suoi URL legittimi sono `https://`, lunghi al massimo
58 caratteri, senza percorso, mentre quelli di phishing arrivano a 6.097 caratteri. Un modello
addestrato lì impara la lunghezza.

| fonte | ruolo | che cos'è |
|---|---|---|
| Phishing.Database (lista ACTIVE) | phishing | URL di phishing attivi, aggiornati ogni giorno |
| Wikipedia, link esterni | benigni | pagine realmente citate dalle voci: percorsi veri, domini diversissimi |
| Wikipedia, ricerca per TLD | benigni | stessa fonte interrogata TLD per TLD, per coprire `.io .ai .dev .xyz` |
| Sitemap di siti Tranco | benigni | pagine profonde di siti veri, anche su TLD moderni |
| Wikipedia IT / DE / FR / ES | benigni | perché il web buono non è tutto in inglese |
| **Pagine di accesso verificate** | benigni | 3.941 pagine `/login`, `/account`, `/checkout` che hanno risposto HTTP 200 |
| Homepage Tranco | benigni | campione lungo tutta la classifica, non solo i siti famosi |
| OpenPhish · Hacker News · lobste.rs · URLhaus | **tenuti fuori** | mai usati in addestramento |

Regole di composizione, ognuna nata da un errore vero:

- massimo 3 URL per dominio registrato, altrimenti il modello memorizza gli host compromessi;
- ogni TLD usato dal phishing deve avere anche dei benigni;
- benigni di forme diverse (homepage, pagina profonda, pagina di accesso);
- **le pagine di accesso entrano tutte, e ripetute, fino al 7% dei benigni.** Sono poche ma
  sono la categoria che il phishing imita: alla loro frequenza naturale (1%) sparivano nel
  campione e il modello tornava a bollare il login di chiunque.

## Limiti dichiarati

- **Siti legittimi compromessi.** Quando la truffa vive dentro un sito vero bucato, nella
  stringa non c'è niente da vedere. È il tetto del problema, non del modello.
- **Dizionario inglese.** Le due misure di leggibilità usano una lista di parole inglesi,
  quindi un dominio in un'altra lingua parte svantaggiato. I dati benigni sono multilingua, il
  dizionario non ancora.
- **Nessun segnale di reputazione.** Il modello non sa l'età di un dominio, né il suo
  certificato, né la reputazione dell'IP. In produzione questi si affiancano e alzano molto il
  risultato.
- **Un falso positivo noto**: `accounts.google.com/signin` viene ancora dato per phishing. La
  cura è più pagine di accesso dei grandi provider fra i benigni: lavoro misurabile, non un
  mistero.
- **Un modello di un giorno.** Il phishing si muove. Un modello fermo invecchia: il
  riaddestramento fa parte del progetto, non è un ripensamento.

## Riprodurre, e la licenza dei dati

I dati raccolti **non** stanno in questo repository, apposta: il feed gratuito di OpenPhish
vieta la ridistribuzione, Tranco chiede di non ridistribuire la lista e Phishing.Database ha
la sua licenza. Quello che si pubblica sono i raccoglitori, così chiunque ricostruisce il
dataset dalle fonti vive. Questo rende il progetto riproducibile invece che un dump congelato,
e significa che la tua ricostruzione sarà un po' diversa dalla mia, perché i feed cambiano
ogni giorno.

## File

```
02-phishing-url/
  collect_data.py     scarica le fonti pubbliche (cache in data/raw/)
  build_dataset.py    compone il dataset togliendo le scorciatoie   [--naturale]
  build_holdout.py    rifa' solo l'holdout indipendente
  features.py         le feature dalla sola stringa (usate anche in produzione)
  test_features.py    controlli minimi: cadono se l'estrazione si rompe
  train_phishing.py   7 modelli x 2 split, metriche e figure        [--naturale] [--prova N]
  riepilogo.py        tutti i numeri finali in un posto solo
  check_url.py        la riga di comando da usare davvero
  prove_reali.py      prova di accettazione su URL veri e conosciuti
  DIARIO.md           gli otto inganni, e le due correzioni di troppo
  figure/             le figure dell'ultimo run
```

## Un secondo caso studio, stesso principio

[`01-lead-scoring/`](01-lead-scoring/) — 41.188 telefonate reali di una banca (UCI Bank
Marketing). Cinque modelli con confusion matrix. Anche lì il risultato interessante non è
l'AUC: rifacendo lo split in ordine cronologico invece che a caso, il gradient boosting crolla
da 0,814 a 0,644 mentre la semplice regressione logistica regge a 0,741. Il modello che vince
il benchmark non è quello che sopravvive al tempo.

---

Dati: Phishing.Database, OpenPhish, URLhaus (abuse.ch), Tranco, Wikipedia, Hacker News,
lobste.rs. Codice sotto licenza MIT — vedi [LICENSE](LICENSE), inclusa la nota sui dati.

**Nessuno degli URL mostrati in questo repository è un invito a visitarli.**
