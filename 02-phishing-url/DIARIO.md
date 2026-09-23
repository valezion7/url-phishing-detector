# Diario: otto volte che il modello mi ha ingannato (e due correzioni di troppo)

*[Read in English](DIARY.md)*

Ordine cronologico degli errori veri incontrati costruendo il rilevatore di phishing.
Ogni voce ha: cosa sembrava, cosa era davvero, come si è visto, cosa è cambiato.

Nessuno di questi problemi è stato segnalato dal test interno.

---

### 1. Il dataset famoso era inutilizzabile

**Sembrava**: PhiUSIIL (UCI, 2024, 235.795 URL) è il riferimento accademico per questo
compito, e in giro si leggono accuratezze del 99%.

**Era**: tutti i 134.850 URL legittimi sono `https://`, lunghi al massimo 58 caratteri e
senza percorso; i phishing arrivano a 6.097 caratteri. Un modello li separa guardando la
lunghezza.

**Visto con**: due righe di `groupby(label)` su lunghezza e schema, prima di addestrare.

**Fatto**: buttato. Dati costruiti da zero da feed vivi (Phishing.Database, OpenPhish,
URLhaus) più fonti benigne indipendenti fra loro.

### 2. Lo slash finale

**Sembrava**: AUC 0,998 sullo split per dominio. Sembrava finita.

**Era**: le homepage benigne le avevo generate io come `https://dominio/`, con lo slash.
I link veri di Hacker News non ce l'hanno. Il modello aveva imparato «finisce con slash,
allora è buono»: `https://www.wbeuvvfx.com/` (phishing) → 0,000;
`https://3dassetstudio.com` (sito vero) → 1,000.

**Visto con**: l'holdout indipendente. Un quarto dei link buoni veniva bloccato.

**Fatto**: canonicalizzazione. Via lo slash finale della home e il prefisso `www.`: non
dicono niente sulla pericolosità, dicono da quale lista viene l'URL.

### 3. Lo schema http/https

**Sembrava**: feature ovvia e gratis, i siti seri sono in HTTPS.

**Era**: i benigni venivano da citazioni di Wikipedia, piene di vecchi link `http://`.
Nel dataset `http` voleva dire «buono»: l'opposto della realtà. 56% di https fra i buoni
contro 74% fra i cattivi.

**Fatto**: schema escluso dal modello. Resta solo nei report.

### 4. Nessun TLD moderno fra i buoni

**Sembrava**: il modello penalizza `.xyz`, `.online`, `.top`. Ragionevole: sono TLD abusati.

**Era**: fra i benigni c'erano **zero** `.ai`, **zero** `.io`, **zero** `.dev`, mentre su
Hacker News sono il 15% dei link. Qualunque sito su un TLD moderno veniva bloccato:
`ageofinvention.xyz`, `cactuscompute.com`, `supabase.link`.

**Visto con**: tabella dei TLD per classe, affiancata a quella dell'holdout.

**Fatto**: 162.641 URL benigni raccolti TLD per TLD dalle citazioni di Wikipedia, più un
crawl di sitemap mirato ai domini `.io .ai .dev .xyz .app ...`.

### 5. Il campione dei benigni prendeva solo i famosi

**Sembrava**: campionamento log-uniforme sulla classifica Tranco, coda lunga inclusa.

**Era**: un `head()` al posto di un `sample()`. Gli indici erano ordinati, quindi le
homepage benigne stavano tutte fra i primi 15.000 siti del mondo. Il modello aveva imparato
«dominio che non conosco, allora è phishing» e bocciava ogni progetto indie
(`un-scrabbled.com`, `a0flow.com`, `proudsend.com`).

**Fatto**: campione vero sulla coda lunga, fino al rango 600.000.

### 6. La lunghezza non distingue un sito piccolo da un dominio usa e getta

**Sembrava**: bastavano le feature di forma (lunghezze, cifre, sottodomini).

**Era**: `proudsend.com` e `webufexkp.com` hanno la stessa forma. Quello che cambia è se il
nome **si legge**.

**Fatto**: due feature nuove sul nome a dominio — quanta parte è fatta di parole vere
(segmentazione su un dizionario) e quanto i bigrammi somigliano a una lingua.

| nome | copertura in parole | bigrammi |
|---|---|---|
| `proudsend` | 1,00 | −3,6 |
| `3dassetstudio` | 0,85 | −4,2 |
| `un-scrabbled` | 0,82 | −3,5 |
| `webufexkp` | 0,33 | −6,0 |
| `wbeuvvfx` | 0,00 | −8,0 |

### 7. Le pagine di accesso vere mancavano

**Sembrava**: il modello riconosce bene `login`, `verify`, `account`.

**Era**: fra i benigni non c'era quasi nessuna pagina di accesso — Wikipedia e le sitemap
non le contengono. Il modello aveva imparato «percorso di login, allora è phishing» e dava
`https://dashboard.<cliente>.com/login`, pannello reale di un cliente, al **99,5%** di
phishing.

**Visto con**: una prova manuale su URL veri e conosciuti. Nessun test automatico l'avrebbe
trovato, perché nei dati quella categoria non esisteva.

**Fatto**: crawl di pagine di accesso vere (`/login`, `/account`, `/checkout`, `/register`…)
verificate con HTTP 200 su migliaia di domini Tranco, tenendo l'URL **finale** dopo i
redirect — che è spesso ancora più simile a un phishing
(`accounts.esempio.com/signin?next=...`).

Non è bastato raccoglierle: a frequenza naturale erano l'1% dei benigni e il modello
continuava a sbagliare (`accounts.google.com/signin` dato per phishing). Entrano tutte
(3.941 da due giri di raccolta), ripetute fino al **7%**. È una scelta esplicita: quella
categoria pesa più di quanto il campionamento le darebbe, perché è esattamente quella su
cui l'errore costa.

Quanto conta si misura: il solo secondo giro di raccolta ha portato l'AUC sull'holdout da
0,940 a **0,948** e il phishing preso all'1% di falsi positivi dal 49% al **53%**.

### 8. Il web non è tutto in inglese

**Sembrava**: sistemate le pagine di accesso, restava solo da addestrare.

**Era**: provando su siti italiani veri, la home di **Intesa Sanpaolo**
(`intesasanpaolo.com/it/persone-e-famiglie.html`) prendeva **0,94**. Tutte le fonti benigne
erano di fatto in inglese — Wikipedia inglese, sitemap di siti globali — mentre il phishing
in italiano nel feed c'è (17.163 URL con parole italiane, 2,2% del totale). Risultato: le
parole italiane in un URL spingevano verso «sospetto».

**Visto con**: una manciata di URL italiani conosciuti, messi alla prova a mano. Nessuna
metrica aggregata lo mostrava: in quel test set l'italiano quasi non esisteva.

**Fatto**: link esterni raccolti anche dalle Wikipedia italiana, tedesca, francese e
spagnola. E `prove_reali.py`, che tiene la prova come controllo ripetibile.

---

## Due correzioni di troppo

Le due volte in cui ho *migliorato* il modello e l'ho peggiorato. Entrambe si vedono solo
da fuori: il test interno diceva che andava meglio.

### a. Pareggiare TLD e forma

Visto che TLD e forma dell'URL portavano scorciatoie, la mossa apparentemente rigorosa era
pareggiare: per ogni combinazione (TLD, ha un percorso sì/no), tanti benigni quanti phishing.
Così quelle due variabili, da sole, non dicono più niente.

L'AUC interna è scesa da 0,986 a 0,961 e sembrava il conto della scorciatoia. Sbagliato:

| | AUC test interno | AUC holdout | phishing preso bloccando l'1% dei buoni |
|---|---|---|---|
| pareggiato (TLD × forma) | 0,960 | 0,912 | 18,0% |
| naturale (solo copertura garantita) | 0,986 | **0,937** | **51,0%** |

I TLD abusati *lo sono davvero*: la distribuzione naturale è informazione vera. Il problema
non era che il TLD portasse segnale, era che fra i benigni certi TLD **non esistevano**. La
correzione giusta è **garantire copertura**, non imporre uguaglianza.

### b. Tre viste testuali invece di una

Sembrava un'ovvia raffinatezza: invece di un solo TF-IDF di caratteri su tutto l'URL, tre
viste separate — caratteri del nome host, caratteri del percorso, parole intere. «login» nel
dominio e «login» nel percorso non vogliono dire la stessa cosa.

| | AUC test interno | AUC holdout | phishing preso bloccando l'1% dei buoni |
|---|---|---|---|
| tre viste separate | **0,9849** | 0,907 | 25,8% |
| una vista sola | 0,9841 | **0,9500** | **53,6%** |

Stessi dati, stesso split, stesso classificatore: cambia solo come viene vista la stringa.
Le viste separate lasciano memorizzare pezzi di nome host visti in addestramento, e **lo
split per dominio non se ne accorge** perché phishing e benigni continuano a venire dalle
stesse fonti. Se ne accorge solo un holdout raccolto altrove.

**La regola che mi porto dietro**: prima di adottare un miglioramento perché alza il numero,
misurarlo su dati esterni. Una scorciatoia e un segnale vero si assomigliano molto; un
raffinamento e un overfitting anche.

---

**La morale operativa**: il test interno non ha mai segnalato niente di tutto questo. Li ha
trovati tutti l'holdout indipendente — dati raccolti da qualcun altro, in un altro modo, lo
stesso giorno — più una prova manuale su URL conosciuti. Senza quelli, oggi avrei in mano un
modello con AUC 0,998 che blocca un quarto del web buono.
