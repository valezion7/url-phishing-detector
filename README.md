# Eight ways a classifier lied to me

*[Leggi in italiano](README.it.md)*

A phishing-URL detector that judges a link from **the string alone** — no network call, no
page fetch — plus the log of every shortcut it took while I was building it, and how each one
was caught.

The point of this repository is not the model. It is that the model's own test said **0.99**
while it was blocking a quarter of the good web, and that the only thing which ever noticed
was a test set someone else had collected.

| | ROC-AUC |
|---|---|
| random split | 0.989 |
| **split by domain** (no domain shared between train and test) | **0.987** |
| **independent holdout** (today's verified phishing vs links people posted today) | **0.948** |

On the independent holdout, blocking **0.9% of good links** it catches **53% of fresh
phishing** (95% CI: 46.6–60.6%) and **93% of URLhaus malware URLs** — a different threat it
never saw in training. Accept a 4.9% false-positive rate and it reaches 78%.

Everything runs on CPU with a stock scientific-Python stack. Full training: about 17 minutes.
Scoring one URL: microseconds.

---

## The part worth reading

**[`02-phishing-url/DIARY.md`](02-phishing-url/DIARY.md)** — eight times the model learned
something that had nothing to do with phishing, and two "improvements" that raised the
internal score while making the real one worse.

A taste:

- It learned that **a URL ending in `/` is safe**, because I had generated the benign
  homepages myself and they all had the trailing slash. Real links posted by people do not.
- It learned that **`http://` means benign**, because my benign URLs came from Wikipedia
  citations, which are full of old `http` links. The exact opposite of reality.
- There was not a single **`.ai`, `.io` or `.dev`** among the benign examples, while 15% of
  Hacker News links are. Every site on a modern TLD got blocked.
- It learned that **an unfamiliar domain is phishing**, because of a `head()` where a
  `sample()` belonged: the benign homepages were all in the world's top 15,000 sites.
- It learned that **a login path means phishing**, because benign login pages barely existed
  in the data. A real client's admin panel scored 99.5% phishing.
- Splitting the text features into three views (host / path / words) raised the internal
  AUC from 0.9841 to 0.9849 and **collapsed the external one from 0.950 to 0.907**.

None of these was flagged by the internal test. Not one.

## What the model actually looks at

Two models combined by a stacking meta-learner:

1. **Logistic regression over character n-grams** (3–5, `char_wb`) of the whole URL.
2. **Gradient boosting over 62 computed features** — lengths, subdomain count, digits,
   entropy, a known brand appearing in the hostname under a domain that is not its own,
   shared-hosting providers, and two measures of how *pronounceable* the domain name is.

That last pair is what separates a small indie site from a throwaway domain, which length
alone cannot do:

| domain | dictionary-word coverage | character bigram plausibility |
|---|---|---|
| `proudsend` | 1.00 | −3.6 |
| `3dassetstudio` | 0.85 | −4.2 |
| `un-scrabbled` | 0.82 | −3.5 |
| `webufexkp` | 0.33 | −6.0 |
| `wbeuvvfx` | 0.00 | −8.0 |

Deliberately **excluded** from the model: the `http`/`https` scheme, the trailing slash and
the `www.` prefix. They say which list a URL came from, not whether it is dangerous.

## Quick start

```bash
pip install scikit-learn pandas numpy matplotlib joblib tldextract requests

cd 02-phishing-url
python collect_data.py             # rebuilds the dataset from public feeds (hours, mostly waiting)
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

`python check_url.py --file list.txt` scores a whole list.

## Results in full

### Internal test, split by domain

20% of *domains* held out entirely, so the model cannot recognise a host it has already seen.

| model | AUC random split | AUC by domain | PR-AUC | caught @1% FPR | @0.1% FPR |
|---|---|---|---|---|---|
| Logistic regression (features) | 0.925 | 0.924 | 0.928 | 43.9% | 21.4% |
| Decision tree (features) | 0.865 | 0.865 | 0.867 | 33.9% | 13.1% |
| Random forest (features) | 0.962 | 0.953 | 0.957 | 58.8% | 32.6% |
| HistGradientBoosting (features) | 0.977 | 0.974 | 0.976 | 72.4% | 50.2% |
| Logistic regression (char n-grams) | 0.987 | 0.984 | 0.986 | 82.7% | 59.9% |
| Soft-voting ensemble | 0.989 | 0.987 | 0.988 | 84.2% | 68.4% |
| **Stacking (shipped)** | **0.989** | **0.987** | **0.989** | **84.4%** | **69.8%** |

### Independent holdout — the one that counts

2,210 links posted the same day by people (Hacker News, lobste.rs) against 193 phishing URLs
verified the same day (OpenPhish). No domain in common with training.

| good links blocked | fresh phishing caught | threshold | URLhaus malware caught |
|---|---|---|---|
| 0.50% | 45.6% | 0.984 | 91.1% |
| **0.90%** | **53.4%** | 0.977 | 92.8% |
| 1.99% | 62.7% | 0.958 | 94.3% |
| 4.89% | 78.2% | 0.844 | 96.3% |
| 9.86% | 86.0% | 0.542 | 97.3% |

With only 193 verified phishing URLs the 53.4% carries a 95% interval of 46.6–60.6%. That is
an honest estimate, not a precise measurement, and the repository says so everywhere.

Calibrating the threshold on half the holdout and measuring on the other half:
**0.58% of good links blocked, 50.0% of phishing caught.**

## Where the data comes from

No pre-packaged dataset. The most-cited academic dataset for this task (PhiUSIIL, 235,795
URLs) is unusable: every one of its legitimate URLs is `https://`, at most 58 characters, with
no path, while its phishing URLs run to 6,097 characters. A model trained on it learns length.

| source | role | what it is |
|---|---|---|
| Phishing.Database (ACTIVE list) | phishing | live phishing URLs, updated daily |
| Wikipedia external links | benign | pages actually cited by articles: real paths, very diverse domains |
| Wikipedia, queried per TLD | benign | the same source asked TLD by TLD, to cover `.io .ai .dev .xyz` |
| Sitemaps of Tranco sites | benign | deep pages of real sites, modern TLDs included |
| Wikipedia IT / DE / FR / ES | benign | because the good web is not all in English |
| **Verified login pages** | benign | 3,941 `/login`, `/account`, `/checkout` pages that answered HTTP 200 |
| Tranco homepages | benign | sampled across the whole ranking, not just famous sites |
| OpenPhish · Hacker News · lobste.rs · URLhaus | **held out** | never used for training |

Composition rules, each one born from a real mistake:

- at most 3 URLs per registered domain, or the model memorises compromised hosts;
- every TLD used by phishing must also have benign examples;
- benign URLs in several shapes (homepage, deep page, login page);
- **login pages all enter, and repeated, up to 7% of the benign set.** They are few but they
  are the category phishing imitates: at their natural 1% frequency they vanish into the
  sample and the model goes back to flagging everybody's login page.

## Honest limits

- **Compromised legitimate sites.** When the scam lives inside a real hacked website, there is
  nothing in the string to see. That is the ceiling of the problem, not of the model.
- **English dictionary.** The two readability measures use an English word list, so a domain
  in another language starts at a disadvantage. The benign data is multilingual; the
  dictionary is not yet.
- **No reputation signals.** The model does not know a domain's age, its certificate or its IP
  reputation. In production those sit alongside it and raise the result a lot.
- **A known false positive**: `accounts.google.com/signin` still scores as phishing. The fix is
  more big-provider login pages among the benign examples — measurable work, not a mystery.
- **A model of one day.** Phishing moves. A frozen model decays; retraining is part of the
  design, not an afterthought.

## Reproducing, and the data licence

The collected data is **not** in this repository, on purpose: the OpenPhish community feed
forbids redistribution, Tranco asks that its list not be redistributed, and Phishing.Database
carries its own licence. What is published is the collectors, so anyone rebuilds the dataset
from the live sources. That makes the project reproducible rather than a frozen dump — and it
means your rebuild will differ slightly from mine, because the feeds move every day.

## Files

```
02-phishing-url/
  collect_data.py     downloads the public sources (cached in data/raw/)
  build_dataset.py    composes the dataset, removing the shortcuts   [--naturale]
  build_holdout.py    rebuilds only the independent holdout
  features.py         the features, from the string alone (also used at inference)
  test_features.py    minimal checks: they fail if extraction breaks
  train_phishing.py   7 models x 2 splits, metrics and figures       [--naturale] [--prova N]
  riepilogo.py        every final number in one place
  check_url.py        the command line you would actually use
  prove_reali.py      acceptance test on real, known URLs
  DIARY.md            the eight deceptions, and the two over-corrections
  figure/             the figures from the last run
```

Script names and console output are in Italian; the documentation is bilingual.

## A second case study, same principle

[`01-lead-scoring/`](01-lead-scoring/) — 41,188 real bank phone calls (UCI Bank Marketing).
Five models with confusion matrices. Again, the interesting result is not the AUC: redo the
split in chronological order instead of at random and gradient boosting falls from 0.814 to
0.644 while plain logistic regression holds at 0.741. The model that wins the benchmark is
not the one that survives time.

---

Data: Phishing.Database, OpenPhish, URLhaus (abuse.ch), Tranco, Wikipedia, Hacker News,
lobste.rs. Code under MIT — see [LICENSE](LICENSE), including the note on data.

**None of the URLs shown in this repository is an invitation to visit them.**
