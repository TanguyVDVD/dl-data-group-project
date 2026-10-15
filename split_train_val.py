"""Répartit 2euro_by_country/ (et mes_photos/ s'il existe) en dataset_split/train et dataset_split/val (80/20 par pays).

- Répartition stratifiée : chaque pays garde la même proportion train/val.
- Seed fixe : la répartition est identique à chaque exécution.
- Doublons exacts ET quasi-doublons (même photo redimensionnée/recompressée, fréquents avec un crawler web)
  supprimés, sinon une même image pourrait se retrouver à la fois dans train et val.
- Quasi-doublons rangés dans des pays différents = erreur d'étiquette dans le dataset d'origine :
  les deux images sont écartées et listées.
- mes_photos/<Pays>/ : vos propres photos (webcam...), ajoutées en priorité.

Ensuite : python download_ecb.py (images BCE) puis python balance_train.py (équilibrage).
"""
import hashlib
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
SOURCES = [ROOT / "mes_photos", ROOT / "2euro_by_country"]  # par ordre de priorité
DST = ROOT / "dataset_split"
VAL_RATIO = 0.2
SEED = 42
NEAR_DUP_BITS = 12  # distance de Hamming max (sur 256 bits) entre deux empreintes pour « même image »
IMG_EXT = {".jpg", ".jpeg", ".png"}


def dhash(path, size=16):
    """Empreinte perceptuelle (256 bits) : proche pour deux versions d'une même photo."""
    g = np.asarray(Image.open(path).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


# 1. inventaire + suppression des doublons exacts
images, seen_md5, n_exact = [], set(), 0
for src in SOURCES:
    if not src.exists():
        continue
    for img in sorted(src.glob("*/*")):
        if img.suffix.lower() not in IMG_EXT:
            continue
        h = hashlib.md5(img.read_bytes()).hexdigest()
        if h in seen_md5:
            n_exact += 1
            continue
        seen_md5.add(h)
        images.append(img)
labels = np.array([p.parent.name for p in images])
print(f"{len(images)} images uniques ({n_exact} doublons exacts ignorés)")

# 2. quasi-doublons
hashes = np.stack([dhash(p) for p in images])
dist = (hashes[:, None, :] != hashes[None, :, :]).sum(2)
np.fill_diagonal(dist, 999)
near = dist <= NEAR_DUP_BITS
conflict = np.array([(labels[near[i]] != labels[i]).any() for i in range(len(images))])
keep = ~conflict
for i in range(len(images)):  # parmi des quasi-doublons de même pays, on garde le premier (priorité des sources)
    if keep[i]:
        keep[np.flatnonzero(near[i])[np.flatnonzero(near[i]) > i]] = False

print(f"{(~keep & ~conflict).sum()} quasi-doublons ignorés")
if conflict.any():
    print(f"{conflict.sum()} images écartées (même image rangée dans des pays différents) :")
    for i in np.flatnonzero(conflict):
        print("   ", images[i].relative_to(ROOT))

# 3. répartition stratifiée
if DST.exists():
    shutil.rmtree(DST)
rng = random.Random(SEED)
total = {"train": 0, "val": 0}
print(f"\n{'pays':15} {'train':>6} {'val':>5}")
for country in sorted(set(labels)):
    files = [images[i] for i in np.flatnonzero(keep & (labels == country))]
    rng.shuffle(files)
    n_val = max(1, round(len(files) * VAL_RATIO))
    for split, subset in {"val": files[:n_val], "train": files[n_val:]}.items():
        out = DST / split / country
        out.mkdir(parents=True, exist_ok=True)
        for img in subset:
            shutil.copy2(img, out / img.name)
        total[split] += len(subset)
    print(f"{country:15} {len(files) - n_val:>6} {n_val:>5}")

print(f"\ntrain {total['train']} | val {total['val']}")
