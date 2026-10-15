"""Construit dataset_euro/ pour le modèle « qu'est-ce que je montre à la caméra ? » (3 classes).

euro  (toutes les valeurs, 1 cent à 2 €)
  - photonsquid/coins-euro (Hugging Face, MIT)  : vraies photos de pièces, de biais, fond texturé
  - EURO-Coin (DAISLab-Unisa, CC BY-NC 4.0)     : N_PER_VALUE images par valeur (1c ... 1 €)
  - 2euro_by_country/ (EURO-Coin 2 €, déjà local) : N_2EURO images
  - mes_photos/<Pays>/ et mes_photos_euro/      : vos photos webcam (mode collecte de camera.py)
piece (une pièce, mais pas en euro)
  - Francesco/coins-1apki (Hugging Face, CC)    : pièces d'autres pays (forint, couronne, won, peso...)
  - Aff4n20/ancient-coin-dataset (Hugging Face) : pièces antiques
  - mes_photos_piece/                            : vos photos de pièces non euro (CHF, £, $...)
objet (pas une pièce)
  - ImageNet (mlnomad/imnet1k_*, Hugging Face)   : bouchons de bouteille (toutes les photos) + objets ronds
                                                   ou petits : cache d'objectif, boucle, boussole, balles...
  - Francesco/coins-1apki                        : vis, écrous, clous
  - Imagenette (fast.ai)                         : objets et scènes variés
  - mes_photos_objet/                            : vos photos d'objets (bouchons, boutons, rondelles, jetons...)

Chaque image est recadrée en carré sur l'objet et réduite à 320 px (comme le recadrage de camera.py).
Doublons et quasi-doublons supprimés, puis répartition 80/20 stratifiée par classe et par source.
Les téléchargements et images préparées sont gardés dans .cache_euro/ : relancer le script est rapide.
"""
import hashlib
import io
import json
import random
import re
import shutil
import tarfile
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).parent
CACHE = ROOT / ".cache_euro"
PREPARED = CACHE / "prepared"
DST = ROOT / "dataset_euro"
VAL_RATIO = 0.2
SEED = 42
SIZE = 320
N_PER_VALUE = 250      # images EURO-Coin par valeur (1c, 2c, 5c, 10c, 20c, 50c, 1 €)
N_2EURO = 300          # images 2 € prises dans 2euro_by_country/
N_IMAGENETTE = 500
# classes ImageNet d'objets ronds ou petits qu'on pourrait montrer à la caméra (50 photos chacune)
IMAGENET_OBJECTS = [
    "lens_cap_lens_cover", "buckle", "magnetic_compass", "stopwatch_stop_watch", "analog_clock", "wall_clock",
    "golf_ball", "ping-pong_ball", "rubber_eraser_rubber_pencil_eraser", "thimble", "hair_slide", "safety_pin",
    "lighter_light_igniter_ignitor", "loupe_jeweler_s_loupe", "combination_lock", "padlock", "nail", "screw",
    "pill_bottle", "mouse_computer_mouse", "remote_control_remote", "coffee_mug", "sundial", "dial_telephone_dial_phone",
]
NEAR_DUP_BITS = 12
IMG_EXT = {".jpg", ".jpeg", ".png"}
HF = "https://huggingface.co"
HEADERS = {"User-Agent": "Mozilla/5.0 (projet etudiant - dataset pieces euro)"}
rng = random.Random(SEED)


def download(url, name):
    """Télécharge une fois dans .cache_euro/ et retourne le chemin local."""
    out = CACHE / name
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(urllib.parse.quote(url, safe=":/?=&%"), headers=HEADERS)
        out.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        time.sleep(0.1)
    return out


def square(img, box=None, margin=0.08):
    """Carré centré sur l'objet `box` (x, y, w, h) avec une marge, ou carré central de l'image.
    Fond blanc si le carré dépasse de l'image. Réduit à SIZE px."""
    img = img.convert("RGB")
    if box is None:
        side, cx, cy = min(img.size), img.width / 2, img.height / 2
    else:
        x, y, w, h = box
        side, cx, cy = max(w, h) * (1 + 2 * margin), x + w / 2, y + h / 2
    left, top = int(cx - side / 2), int(cy - side / 2)
    canvas = Image.new("RGB", (int(side), int(side)), "white")
    canvas.paste(img.crop((max(left, 0), max(top, 0), min(left + int(side), img.width), min(top + int(side), img.height))),
                 (max(-left, 0), max(-top, 0)))
    canvas.thumbnail((SIZE, SIZE), Image.Resampling.LANCZOS)
    return canvas


def prepared(cls, source, key):
    return PREPARED / cls / f"{source}_{key}.jpg"


def save(img, cls, source, key):
    out = prepared(cls, source, key)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, quality=92)


def done(source):
    """Une source déjà préparée lors d'une exécution précédente n'est pas retraitée."""
    flag = PREPARED / f".{source}.done"
    return flag.exists()


def mark_done(source):
    (PREPARED / f".{source}.done").touch()


def hf_parquets(repo):
    info = json.load(urllib.request.urlopen(urllib.request.Request(f"{HF}/api/datasets/{repo}", headers=HEADERS)))
    files = [s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith(".parquet")]
    return [download(f"{HF}/datasets/{repo}/resolve/main/{f}", f"{repo.replace('/', '__')}/{Path(f).name}") for f in files]


def hf_list(repo, folder):
    """Liste tous les fichiers d'un dossier d'un dataset Hugging Face (pagination incluse)."""
    url, files = f"{HF}/api/datasets/{repo}/tree/main/{urllib.parse.quote(folder)}", []
    while url:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS))
        files += [x["path"] for x in json.load(resp) if x["type"] == "file"]
        nxt = re.search(r'<([^>]+)>;\s*rel="next"', resp.headers.get("Link", ""))
        url = nxt.group(1) if nxt else None
    return files


PREPARED.mkdir(parents=True, exist_ok=True)

# ---------- euro ----------
if not done("photonsquid"):
    print("photonsquid/coins-euro ...")
    for f in hf_parquets("photonsquid/coins-euro"):
        for i, row in enumerate(pq.read_table(f).to_pylist()):
            save(square(Image.open(io.BytesIO(row["image"]["bytes"]))), "euro", "photonsquid", f"{f.stem[:5]}{i}")
    mark_done("photonsquid")

if not done("eurocoin"):
    print("EURO-Coin (Hugging Face) ...")
    for value in ["1-cent", "2-cent", "5-cent", "10-cent", "20-cent", "50-cent", "1-euro"]:
        files = hf_list("DAISLab-Unisa/EURO_Coin_Dataset", f"Dataset_Value/{value}")
        for path in rng.sample(files, min(N_PER_VALUE, len(files))):
            if prepared("euro", "eurocoin", Path(path).stem).exists():
                continue
            try:  # l'original n'est pas gardé : seule l'image préparée (320 px) est en cache
                url = urllib.parse.quote(f"{HF}/datasets/DAISLab-Unisa/EURO_Coin_Dataset/resolve/main/{path}", safe=":/")
                data = urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60).read()
                save(square(Image.open(io.BytesIO(data))), "euro", "eurocoin", Path(path).stem)
            except Exception as e:  # image corrompue ou format exotique
                print("   ignorée :", path, e)
        print("  ", value, "ok")
    mark_done("eurocoin")

if not done("2euro"):
    local_2e = sorted(p for p in (ROOT / "2euro_by_country").glob("*/*") if p.suffix.lower() in IMG_EXT)
    for p in rng.sample(local_2e, min(N_2EURO, len(local_2e))):
        save(square(Image.open(p)), "euro", "2euro", p.stem)
    mark_done("2euro")

# ---------- autre ----------
if not done("francesco"):
    print("Francesco/coins-1apki (autres pièces, vis, écrous, clous) ...")
    for f in hf_parquets("Francesco/coins-1apki"):
        for row in pq.read_table(f).to_pylist():
            img = Image.open(io.BytesIO(row["image"]["bytes"]))
            for k, (box, cat) in enumerate(zip(row["objects"]["bbox"], row["objects"]["category"])):
                if box[2] * box[3] < 0.01 * img.width * img.height:  # objet minuscule : inutilisable
                    continue
                cls = "piece" if cat == 1 else "objet"  # 1 = pièce, 2-4 = clou, écrou, vis
                save(square(img, box), cls, "francesco", f"{row['image_id']}_{k}_c{cat}")
    mark_done("francesco")

if not done("ancient"):
    print("Aff4n20/ancient-coin-dataset ...")
    for f in hf_parquets("Aff4n20/ancient-coin-dataset"):
        for i, row in enumerate(pq.read_table(f).to_pylist()):
            save(square(Image.open(io.BytesIO(row["image"]["bytes"]))), "piece", "ancient", str(i))
    mark_done("ancient")

if not done("imagenette"):
    print("Imagenette ...")
    tgz = download("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz", "imagenette2-160.tgz")
    with tarfile.open(tgz) as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".jpeg")]
        for m in rng.sample(members, min(N_IMAGENETTE, len(members))):
            save(square(Image.open(tar.extractfile(m))), "objet", "imagenette", Path(m.name).stem)
    mark_done("imagenette")

if not done("imagenet"):
    print("ImageNet : bouchons de bouteille + objets ronds ...")
    repo_files = {"bottlecap": ["train", "validation"], **{name: ["validation"] for name in IMAGENET_OBJECTS}}
    for name, splits in repo_files.items():
        for split in splits:
            f = download(f"{HF}/datasets/mlnomad/imnet1k_{name}/resolve/main/data/{split}-00000-of-00001.parquet",
                         f"imagenet/{name}_{split}.parquet")
            for i, row in enumerate(pq.read_table(f, columns=["image"]).to_pylist()):
                save(square(Image.open(io.BytesIO(row["image"]["bytes"]))), "objet", "imagenet", f"{name[:12]}-{split[:3]}{i}")
        print("  ", name, "ok")
    mark_done("imagenet")

# vos photos (toujours relues : elles changent)
own = {"euro": [*(ROOT / "mes_photos").glob("*/*"), *(ROOT / "mes_photos_euro").glob("*")],
       "piece": [*(ROOT / "mes_photos_piece").glob("*")],
       "objet": [*(ROOT / "mes_photos_objet").glob("*")]}
for cls, paths in own.items():
    for old in PREPARED.glob(f"{cls}/photo_*"):
        old.unlink()
    for p in paths:
        if p.suffix.lower() in IMG_EXT:
            save(square(Image.open(p)), cls, "photo", f"{p.parent.name}_{p.stem}")

# ---------- doublons ----------
items = sorted(p for p in PREPARED.glob("*/*.jpg"))
# priorité à vos photos puis aux vraies photos photonsquid quand deux images sont identiques
priority = {"photo": 0, "photonsquid": 1}
items.sort(key=lambda p: priority.get(p.stem.split("_")[0], 9))
labels = np.array([p.parent.name for p in items])
sources = np.array([p.stem.split("_")[0] for p in items])


def dhash(path, size=16):
    g = np.asarray(Image.open(path).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten()).view(np.uint64)


print(f"\nEmpreintes de {len(items)} images ...")
H = np.stack([dhash(p) for p in items])
md5 = [hashlib.md5(p.read_bytes()).hexdigest() for p in items]
keep = np.ones(len(items), bool)
conflict = np.zeros(len(items), bool)
seen_md5 = {}
for i, h in enumerate(md5):
    if h in seen_md5:
        keep[i] = False
    seen_md5.setdefault(h, i)
for start in range(0, len(items), 512):  # comparaison par blocs (mémoire)
    block = np.bitwise_count(H[start:start + 512, None, :] ^ H[None, :, :]).sum(2)
    for r, row in enumerate(block):
        i = start + r
        near = np.flatnonzero(row <= NEAR_DUP_BITS)
        near = near[near != i]
        if (labels[near] != labels[i]).any():
            conflict[i] = True
        elif keep[i]:
            keep[near[near > i]] = False
keep &= ~conflict
print(f"{(~keep).sum()} images retirées ({conflict.sum()} présentes dans deux classes différentes)")

# images mal étiquetées repérées à la main (voir exclusions_euro.txt et la cellule "Nettoyage" de model-euro.ipynb)
excl_file = ROOT / "exclusions_euro.txt"
excluded = {l.strip() for l in excl_file.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")} if excl_file.exists() else set()
is_excluded = np.array([p.stem in excluded for p in items])
keep &= ~is_excluded
print(f"{is_excluded.sum()} images exclues par exclusions_euro.txt")

# ---------- répartition ----------
if DST.exists():
    shutil.rmtree(DST)
counts = Counter()
CLASSES = ("euro", "piece", "objet")
for cls in CLASSES:
    for src in sorted(set(sources[labels == cls])):
        files = [items[i] for i in np.flatnonzero(keep & (labels == cls) & (sources == src))]
        rng.shuffle(files)
        n_val = round(len(files) * VAL_RATIO)
        for k, p in enumerate(files):
            split = "val" if k < n_val else "train"
            out = DST / split / cls / p.name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
            counts[(split, cls, src)] += 1

print(f"\n{'classe':6} {'source':12} {'train':>6} {'val':>5}")
for cls in CLASSES:
    for src in sorted({s for _, c, s in counts if c == cls}):
        print(f"{cls:6} {src:12} {counts[('train', cls, src)]:>6} {counts[('val', cls, src)]:>5}")
for split in ("train", "val"):
    print(split + " : " + " | ".join(f"{cls} {sum(v for (s, c, _), v in counts.items() if s == split and c == cls)}"
                                     for cls in CLASSES))
