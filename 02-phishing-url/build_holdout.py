"""Holdout indipendente: raccolto da fonti che NON hanno addestrato niente, e ripulito
di ogni dominio che compare in uno qualsiasi dei dataset di training."""
import pathlib, pandas as pd
import importlib.util as _u
_sp = _u.spec_from_file_location('_bd', pathlib.Path(__file__).parent / 'build_dataset.py')
# carico solo le funzioni di build_dataset senza rieseguirlo
import types, ast, sys
_src = (pathlib.Path(__file__).parent / 'build_dataset.py').read_text(encoding='utf-8')
_tree = ast.parse(_src)
_keep = [n for n in _tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign))][:14]
B = types.ModuleType('_bd'); B.__file__ = str(pathlib.Path(__file__).parent / 'build_dataset.py')
exec(compile(ast.Module(body=_keep, type_ignores=[]), 'build_dataset.py', 'exec'), B.__dict__)

ROOT = pathlib.Path(__file__).parent
usati = set()
for f in ("dataset.csv", "dataset_naturale.csv"):
    p = ROOT / "data" / f
    if p.exists(): usati |= set(pd.read_csv(p, usecols=["registered_domain"]).registered_domain)

parti = [("openphish.txt", "openphish", 1), ("urlhaus.txt", "urlhaus", 1),
         ("hackernews.txt", "hackernews", 0), ("hackernews_recenti.txt", "hackernews_recenti", 0),
         ("hackernews_storico.txt", "hackernews_storico", 0),
         ("lobsters.txt", "lobsters", 0)]
h = pd.concat([B.norm(B.load(f), s).assign(label=l) for f, s, l in parti], ignore_index=True)
h = h.drop_duplicates("url")
prima = len(h)
h = h[~h.registered_domain.isin(usati)]
h.to_csv(ROOT / "data/holdout_fresh.csv", index=False)
print(f"holdout {len(h):,} (tolti {prima-len(h):,} url su domini gia' visti in training)")
print(h.groupby(["label", "source"]).size().to_string())
