"""Ajoute à dataset_split/ les images officielles des pièces de 2 € publiées par la BCE.

- faces nationales courantes (toutes les séries : ex. Belgique Albert II / Philippe)
- pièces commémoratives de 2004 à aujourd'hui

Répartition stratifiée : chaque pays garde 20 % de ses images (originales + BCE) en val.
À lancer juste après split_train_val.py, puis lancer balance_train.py.
Le script retire d'abord les anciennes images BCE et les copies d'équilibrage : on peut le relancer.
Les téléchargements sont gardés dans .cache_ecb/ : relancer le script ne retélécharge rien.

Les images quasi identiques à une image déjà présente dans dataset_split/ sont ignorées
(sinon une même pièce pourrait se retrouver dans train ET val, ce qui fausse la validation).
Les émissions communes ("Euro area countries") et les pays absents du dataset (Croatie...) sont ignorés.
"""
import hashlib
import io
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
SPLIT = ROOT / "dataset_split"
VAL_RATIO = 0.2
SEED = 42
BASE = "https://www.ecb.europa.eu"
HEADERS = {"User-Agent": "Mozilla/5.0 (projet etudiant - dataset pieces 2 euros)"}
DELAY = 0.5           # secondes entre deux requêtes, pour ne pas surcharger le site
NEAR_DUP_BITS = 12    # distance de Hamming max (sur 256 bits) pour considérer deux images identiques
CACHE = ROOT / ".cache_ecb"

# code BCE -> nom du dossier dans dataset_split/
COUNTRIES = {
    "ad": "Andorra", "at": "Austria", "be": "Belgium", "cy": "Cyprus", "et": "Estonia", "fi": "Finland",
    "fr": "France", "de": "Germany", "gr": "Greece", "ie": "Ireland", "it": "Italy", "lv": "Latvia",
    "lt": "Lithuania", "lu": "Luxembourg", "mt": "Malta", "mo": "Monaco", "nl": "Netherlands",
    "pt": "Portugal", "sm": "San-Marino", "sk": "Slovakia", "sl": "Slovenia", "es": "Spain", "va": "Vatican-City",
}
# nom affiché sur les pages commémoratives -> dossier
COMM_NAMES = {name.replace("-", " ").lower(): name for name in COUNTRIES.values()}
COMM_NAMES |= {"vatican": "Vatican-City", "the netherlands": "Netherlands"}


def get(url, cache=False):
    cached = CACHE / hashlib.md5(url.encode()).hexdigest()
    if cache and cached.exists():
        return cached.read_bytes()
    time.sleep(DELAY)
    quoted = urllib.parse.quote(url, safe=":/")  # certains noms de fichiers contiennent des espaces
    data = urllib.request.urlopen(urllib.request.Request(quoted, headers=HEADERS), timeout=30).read()
    if cache:
        CACHE.mkdir(exist_ok=True)
        cached.write_bytes(data)
    return data


def dhash(img, size=16):
    """Empreinte perceptuelle (256 bits) : proche pour deux versions d'une même photo."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


print("Empreintes des images existantes...")
for old in [*SPLIT.glob("*/*/ecb_*"), *SPLIT.glob("train/*/dup*_*")]:
    old.unlink()
existing = [p for p in SPLIT.glob("*/*/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
hashes = [dhash(Image.open(p)) for p in existing]
md5s = {hashlib.md5(p.read_bytes()).hexdigest() for p in existing}

candidates = {c: [] for c in COUNTRIES.values()}  # pays -> [(nom de fichier, données)]
skipped, ignored = Counter(), Counter()


def add(url, country, name):
    data = get(url, cache=True)
    h = dhash(Image.open(io.BytesIO(data)))
    if hashlib.md5(data).hexdigest() in md5s or (np.stack(hashes) != h).sum(1).min() <= NEAR_DUP_BITS:
        skipped[country] += 1
        return
    hashes.append(h)
    candidates[country].append((name, data))


# 1. faces nationales courantes
for code, country in COUNTRIES.items():
    try:
        html = get(f"{BASE}/euro/coins/html/{code}.en.html").decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        print(f"{country}: page introuvable ({e.code})")
        continue
    for src in sorted(set(re.findall(r'src="(/euro/coins/common/shared/img/[^"]+?_2euro[^"]*\.jpg)"', html, re.I))):
        add(BASE + src, country, "ecb_" + Path(src).name)

# 2. pièces commémoratives, année par année
for year in range(2004, date.today().year + 1):
    try:
        html = get(f"{BASE}/euro/coins/comm/html/comm_{year}.en.html").decode("utf-8", "ignore")
    except urllib.error.HTTPError:
        continue  # année pas encore publiée
    for src, label in re.findall(rf'src="(comm_{year}/[^"]+\.(?:jpg|png))".*?<h3>\s*([^<]+?)\s*<', html, re.S):
        country = COMM_NAMES.get(label.lower())
        if country is None:
            ignored[label] += 1
            continue
        add(f"{BASE}/euro/coins/comm/html/{src}", country, f"ecb_comm_{year}_{Path(src).name.replace(' ', '_')}")

# 3. répartition stratifiée : chaque pays garde VAL_RATIO de ses images (originales + BCE) en val
rng = random.Random(SEED)
print(f"\n{'pays':15} {'+train':>7} {'+val':>5} {'doublons':>8} {'% val':>6}")
for country, items in sorted(candidates.items()):
    n_train = len(list((SPLIT / "train" / country).iterdir()))
    n_val = len(list((SPLIT / "val" / country).iterdir()))
    total = n_train + n_val + len(items)
    to_val = min(len(items), max(0, round(total * VAL_RATIO) - n_val))
    items = sorted(items)
    rng.shuffle(items)
    for k, (name, data) in enumerate(items):
        out = SPLIT / ("val" if k < to_val else "train") / country / name
        out.write_bytes(data)
    print(f"{country:15} {len(items) - to_val:>7} {to_val:>5} {skipped[country]:>8} {(n_val + to_val) / total:>6.1%}")

print(f"\ntotal ajoutées {sum(len(v) for v in candidates.values())} | doublons ignorés {sum(skipped.values())}"
      f" | hors classes : {dict(ignored)}")
print("Ensuite : python balance_train.py")
