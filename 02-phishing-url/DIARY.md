# Diary: eight times the model fooled me (and two corrections too many)

*[Leggi in italiano](DIARIO.md)*

The real mistakes, in the order I hit them, while building the phishing-URL detector.
Each entry has: what it looked like, what it actually was, how it surfaced, what changed.

**Not one of these was flagged by the internal test.**

---

### 1. The famous dataset was unusable

**Looked like**: PhiUSIIL (UCI, 2024, 235,795 URLs) is the academic reference for this task,
and you read 99% accuracies about it everywhere.

**Actually was**: all 134,850 legitimate URLs are `https://`, at most 58 characters long and
without a path; the phishing ones run to 6,097 characters. A model separates them by length.

**Surfaced by**: two lines of `groupby(label)` on length and scheme, before training anything.

**Fix**: thrown away. Data built from scratch out of live feeds (Phishing.Database, OpenPhish,
URLhaus) plus benign sources independent of each other.

### 2. The trailing slash

**Looked like**: AUC 0.998 on the split by domain. Looked finished.

**Actually was**: I had generated the benign homepages myself, as `https://domain/`, with the
slash. Real links posted by people do not have it. The model had learned *"ends in a slash,
therefore safe"*: `https://www.wbeuvvfx.com/` (phishing) → 0.000;
`https://3dassetstudio.com` (a real site) → 1.000.

**Surfaced by**: the independent holdout. A quarter of the good links were being blocked.

**Fix**: canonicalisation. Drop the homepage's trailing slash and the `www.` prefix: they say
nothing about danger, they say which list the URL came from.

### 3. The http/https scheme

**Looked like**: an obvious free feature — serious sites are on HTTPS.

**Actually was**: my benign URLs came from Wikipedia citations, full of old `http://` links.
In my dataset `http` meant *benign*: the opposite of reality. 56% HTTPS among the good ones
against 74% among the bad ones.

**Fix**: the scheme is excluded from the model. It survives only in the reports.

### 4. Not a single modern TLD among the benign examples

**Looked like**: the model penalises `.xyz`, `.online`, `.top`. Reasonable — those are abused
TLDs.

**Actually was**: among the benign examples there were **zero** `.ai`, **zero** `.io`, **zero**
`.dev`, while on Hacker News they are 15% of the links. Every site on a modern TLD got
blocked: `ageofinvention.xyz`, `cactuscompute.com`, `supabase.link`.

**Surfaced by**: a table of TLDs per class, put side by side with the holdout's.

**Fix**: 162,641 benign URLs collected TLD by TLD from Wikipedia citations, plus a sitemap
crawl aimed at `.io .ai .dev .xyz .app ...` domains.

### 5. The benign sample only took the famous sites

**Looked like**: log-uniform sampling across the Tranco ranking, long tail included.

**Actually was**: a `head()` where a `sample()` belonged. The indices were sorted, so the
benign homepages were all inside the world's top 15,000 sites. The model had learned *"a
domain I do not know, therefore phishing"* and failed every indie project
(`un-scrabbled.com`, `a0flow.com`, `proudsend.com`).

**Fix**: a real sample across the long tail, down to rank 600,000.

### 6. Length does not separate a small site from a throwaway domain

**Looked like**: the shape features were enough (lengths, digits, subdomains).

**Actually was**: `proudsend.com` and `webufexkp.com` have the same shape. What differs is
whether the name **reads**.

**Fix**: two new features on the domain name — how much of it is made of real words (greedy
segmentation against a dictionary) and how closely its character bigrams resemble a language.

| name | dictionary-word coverage | bigram plausibility |
|---|---|---|
| `proudsend` | 1.00 | −3.6 |
| `3dassetstudio` | 0.85 | −4.2 |
| `un-scrabbled` | 0.82 | −3.5 |
| `webufexkp` | 0.33 | −6.0 |
| `wbeuvvfx` | 0.00 | −8.0 |

### 7. Real login pages were missing

**Looked like**: the model recognises `login`, `verify`, `account` nicely.

**Actually was**: there was almost no login page among the benign examples — Wikipedia and
sitemaps do not contain them. The model had learned *"a login path, therefore phishing"* and
gave a real client's admin panel, `https://dashboard.<client>.com/login`, **99.5%** phishing.

**Surfaced by**: a manual check on real, known URLs. No automated test would have found it,
because that category did not exist in the data.

**Fix**: a crawl of real login pages (`/login`, `/account`, `/checkout`, `/register`…)
verified with HTTP 200 across thousands of Tranco domains, keeping the **final** URL after
redirects — which is often even more phishing-shaped
(`accounts.example.com/signin?next=...`).

Collecting them was not enough: at their natural frequency they were 1% of the benign set and
the model kept getting it wrong (`accounts.google.com/signin` scored as phishing). They all
enter (3,941 from two collection rounds), repeated, up to **7%**. That is a deliberate choice:
the category weighs more than sampling would give it, because it is exactly where the error
costs.

How much it is worth is measurable: the second collection round alone moved the holdout AUC
from 0.940 to **0.948** and the phishing caught at 1% false positives from 49% to **53%**.

### 8. The web is not all in English

**Looked like**: with the login pages sorted out, all that was left was training.

**Actually was**: testing on real Italian sites, the home page of **Intesa Sanpaolo**
(`intesasanpaolo.com/it/persone-e-famiglie.html`) scored **0.94**. Every benign source was
effectively English — English Wikipedia, sitemaps of global sites — while Italian phishing
does exist in the feed (17,163 URLs with Italian words, 2.2% of the total). The result:
Italian words in a URL pushed towards "suspicious".

**Surfaced by**: a handful of known Italian URLs, tried by hand. No aggregate metric showed
it: in that test set Italian barely existed.

**Fix**: external links collected from the Italian, German, French and Spanish Wikipedias too.
And `prove_reali.py`, which keeps the check as a repeatable test.

---

## Two corrections too many

The two times I *improved* the model and made it worse. Both are only visible from outside:
the internal test said it was going better.

### a. Balancing TLD and URL shape

Since TLD and URL shape were carrying shortcuts, the apparently rigorous move was to balance
them: for every combination of (TLD, has a path yes/no), as many benign as phishing. That way
those two variables, on their own, say nothing any more.

The internal AUC dropped from 0.986 to 0.961 and it looked like the price of the shortcut.
Wrong:

| | internal AUC | holdout AUC | phishing caught at 1% FPR |
|---|---|---|---|
| balanced (TLD × shape) | 0.960 | 0.912 | 18.0% |
| natural (coverage guaranteed only) | 0.986 | **0.937** | **51.0%** |

Abused TLDs *really are* abused: the natural distribution is real information. The problem was
never that the TLD carried signal, it was that certain TLDs **did not exist** among the benign
examples. The right correction is to **guarantee coverage**, not to impose equality.

### b. Three text views instead of one

It looked like an obvious refinement: instead of a single character TF-IDF over the whole URL,
three separate views — characters of the hostname, characters of the path, whole words.
"login" in the domain and "login" in the path do not mean the same thing.

| | internal AUC | holdout AUC | phishing caught at 1% FPR |
|---|---|---|---|
| three separate views | **0.9849** | 0.907 | 25.8% |
| a single view | 0.9841 | **0.9500** | **53.6%** |

Same data, same split, same classifier: only how the string is seen changes. The separate
views let the model memorise fragments of hostnames seen in training, and **the split by
domain does not notice**, because phishing and benign examples still come from the same
sources. Only a holdout collected elsewhere notices.

**The rule I keep**: before adopting an improvement because it raises the number, measure it
on external data. A shortcut and a real signal look very much alike; so do a refinement and an
overfit.

---

**The operational moral**: the internal test never flagged any of this. All of it was found by
the independent holdout — data collected by someone else, in another way, on the same day —
plus a manual check on URLs I know. Without those, I would be holding a model with AUC 0.998
that blocks a quarter of the good web.
