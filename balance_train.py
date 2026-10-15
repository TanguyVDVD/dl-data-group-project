"""Équilibre dataset_split/train : chaque pays est complété par des copies de ses propres images
jusqu'au nombre d'images du pays le mieux fourni (dans la limite de MAX_FACTOR fois l'original).

Les copies (préfixe "dup") ne sont PAS identiques pour le modèle : à chaque epoch, chaque image
reçoit une rotation, un recadrage et un éclairage aléatoires différents.
dataset_split/val n'est pas modifié (la validation doit refléter les vraies proportions).

Relancer le script supprime d'abord les anciennes copies.
"""
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
TRAIN = ROOT / "dataset_split" / "train"
MAX_FACTOR = 3  # un pays n'est jamais multiplié par plus que ça (sinon le modèle voit trop souvent les mêmes pièces)
SEED = 42

rng = random.Random(SEED)
for old in TRAIN.glob("*/dup*_*"):
    old.unlink()

originals = {d.name: sorted(d.iterdir()) for d in sorted(TRAIN.iterdir()) if d.is_dir()}
target = max(len(files) for files in originals.values())
print(f"objectif : {target} images par pays\n{'pays':15} {'avant':>6} {'après':>6}")

for country, files in originals.items():
    n_extra = min(target, len(files) * MAX_FACTOR) - len(files)
    pool = []  # chaque image est copiée le même nombre de fois (±1)
    while len(pool) < n_extra:
        pool += rng.sample(files, len(files))
    for k, src in enumerate(pool[:n_extra]):
        shutil.copy2(src, src.parent / f"dup{k}_{src.name}")
    print(f"{country:15} {len(files):>6} {len(files) + n_extra:>6}")
