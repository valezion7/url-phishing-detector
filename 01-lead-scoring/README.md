# Lead scoring — chi risponde "sì" alla telefonata?

*Documentazione in italiano. English documentation: [repository README](../README.md).*


Caso studio di classificazione su dati reali: **UCI Bank Marketing**, 41.188 telefonate
di una banca portoghese (mag 2008 – nov 2010), obiettivo binario = il cliente ha
sottoscritto il deposito vincolato. Positivi: 11,3%.

Tutto gira su CPU con lo stack Anaconda già installato. Nessuna GPU, ~1 minuto di training.

## Come si lancia

```bash
"C:/ProgramData/anaconda3/python.exe" train.py            # addestra i 5 modelli, scrive outputs/
"C:/ProgramData/anaconda3/python.exe" eval_time_split.py  # controllo onesto: split cronologico
"C:/ProgramData/anaconda3/python.exe" score.py lista.csv  # ordina una lista di contatti
```

## Scelte che contano

- **`duration` rimossa.** È la durata della chiamata: si conosce solo *dopo* l'esito.
  Tenendola l'AUC sale a ~0,93 ed è un numero falso — è il motivo per cui in giro si
  leggono risultati gonfiati su questo dataset.
- **`class_weight="balanced"`** ovunque possibile: con 11% di positivi l'accuracy non
  dice niente (il modello "no" a tutti fa 88,7%).
- **Metrica di business: recall sul top 10%.** Se il call center chiama solo i primi
  10% della lista, quanti dei sì effettivi prende?

## Risultati — split casuale 80/20 (test: 8.238 contatti)

| modello | ROC-AUC | PR-AUC | CV ROC-AUC (5-fold) | recall @ top10% |
|---|---|---|---|---|
| Logistic Regression | 0,801 | 0,460 | 0,790 | 45,3% |
| Decision Tree (depth 5) | 0,799 | 0,438 | 0,779 | 46,3% |
| Random Forest (400) | 0,808 | **0,493** | 0,789 | **48,9%** |
| HistGradientBoosting | **0,814** | **0,493** | 0,798 | 47,0% |
| Voting ensemble (soft) | 0,810 | 0,491 | 0,796 | 47,6% |

Baseline: chiamare a caso il 10% della lista → 10% dei sì. Il modello ne prende ~4,8×.

Confusion matrix, curve ROC/PR, albero leggibile e pesi delle feature: cartella `outputs/`.

## Il controllo che ribalta il quadro — split cronologico

Le righe sono in ordine di tempo e tre feature (`euribor3m`, `nr.employed`,
`emp.var.rate`) sono di fatto un orologio: con lo split casuale il modello **vede il
futuro**. Riallenando sul primo 80% del periodo e testando sull'ultimo 20%:

| modello | ROC-AUC (casuale) | ROC-AUC (cronologico) | recall @ top10% |
|---|---|---|---|
| Logistic Regression | 0,801 | **0,741** | 18,3% |
| HistGradientBoosting | 0,814 | 0,644 | 15,0% |

Due cose, entrambe utili:

1. **Il vincitore cambia.** Il boosting, migliore sul benchmark, out-of-time crolla sotto
   la regressione logistica: ha imparato il regime macroeconomico, non il cliente.
2. **Il periodo finale è un altro mondo**: 30,8% di positivi contro 11,3% medio. Il
   lift reale scende da 4,8× a 1,8×.

Togliere le feature macro non salva il boosting (AUC 0,657): il problema non è solo
quella scorciatoia, è che la popolazione cambia.

## Limiti dichiarati

- Dati 2008-2010, mercato portoghese: **non** trasferibili così come sono a un altro Paese o periodo.
- Nessuna feature sensibile è stata rimossa oltre a quelle assenti dal dataset; `age`,
  `job`, `education`, `marital` restano nel modello. Per un uso reale va fatta una
  valutazione di fairness prima del deploy.
- Il modello salvato (`outputs/model.joblib`) è quello dello split casuale: buono per
  la demo, **non** per decidere chi chiamare domani senza riaddestrare sui dati recenti.

## Cosa c'è dentro

```
train.py                 5 modelli, metriche, 4 figure, model.joblib
eval_time_split.py       validazione out-of-time (la parte onesta)
score.py                 CLI: da CSV di contatti a lista ordinata
data/                    dataset UCI (non versionato altrove)
outputs/                 metriche JSON + PNG + modello
```

Dataset: S. Moro, P. Cortez, P. Rita (2014), *A Data-Driven Approach to Predict the Success
of Bank Telemarketing*, Decision Support Systems — UCI ML Repository, CC BY 4.0.
