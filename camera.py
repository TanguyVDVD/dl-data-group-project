"""Reconnaissance en direct du pays d'une pièce de 2 € avec une webcam.

Pipeline pour maximiser la réussite du classifieur :
1. Détection des pièces (cercles de Hough) : le modèle ne voit que la pièce, pas toute la scène.
2. Recadrage carré centré + masque hors du cercle (fond blanc, comme les photos de catalogue du dataset).
3. Option --tta : la pièce est classée dans 4 orientations et les probabilités sont moyennées.
   À activer seulement si le modèle a été entraîné avec des rotations (sinon la précision baisse).
4. Lissage temporel : chaque pièce est suivie d'une image à l'autre et ses probabilités sont moyennées.
5. Seuil de confiance : en dessous, la pièce est affichée comme "incertain".

Mode collecte (k) : photographier ses propres pièces pour enrichir le dataset.
    Choisir le pays avec les flèches gauche/droite (ou a/d), poser UNE pièce, puis :
    - Espace : enregistre la pièce dans mes_photos/<Pays>/
    - r      : rafale on/off (une photo toutes les 0,4 s : tourner la pièce, changer l'éclairage...)
    L'image enregistrée est exactement ce que voit le modèle (recadrage + masque).
    Ensuite : python split_train_val.py && python download_ecb.py && python balance_train.py

Usage :
    python camera.py                          # webcam, poids runs/coin-cls/weights/best.onnx
    python camera.py --camera 0 --weights runs/coin-cls-rot/weights/best.onnx --tta
    python camera.py --collect                # démarre directement en mode collecte
    python camera.py --image photo.jpg        # tester sur une photo, sans webcam

Touches : q quitter | m masque on/off | c mode centre (sans détection) on/off | s sauvegarder les pièces
          k mode collecte on/off | flèches ou a/d changer de pays | Espace photo | r rafale
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default=ROOT / "runs/coin-cls/weights/euro.onnx", type=Path)
parser.add_argument("--camera", default=1, type=int, help="index de la webcam")
parser.add_argument("--image", type=Path, help="classer une photo au lieu de la webcam")
parser.add_argument("--conf", default=0.6, type=float, help="confiance minimale pour afficher un pays")
parser.add_argument("--min-radius", default=40, type=int, help="rayon minimal d'une pièce (px)")
parser.add_argument("--max-radius", default=250, type=int, help="rayon maximal d'une pièce (px)")
parser.add_argument("--max-coins", default=5, type=int, help="nombre maximal de pièces par image")
parser.add_argument("--tta", action="store_true", help="moyenne sur 4 rotations (modèle entraîné avec rotations)")
parser.add_argument("--collect", action="store_true", help="démarrer en mode collecte de photos")
parser.add_argument("--burst", default=0.4, type=float, help="intervalle (s) entre deux photos en rafale")
parser.add_argument("--smooth", default=0.3, type=float, help="poids de la nouvelle image dans la moyenne (0-1)")
parser.add_argument("--hough", default=60, type=int, help="seuil de Hough : plus haut = moins de faux cercles")
parser.add_argument("--min-edge", default=0.6, type=float,
                    help="part minimale du cercle qui doit suivre un vrai contour (0-1)")
parser.add_argument("--max-bg", default=0.5, type=float,
                    help="part maximale de contours juste autour de la pièce (0-1) : rejette les fonds texturés")
parser.add_argument("--confirm", default=4, type=int, help="nombre d'images consécutives avant d'afficher une pièce")
args = parser.parse_args()

model = YOLO(args.weights)
NAMES = model.names
if args.weights.suffix == ".onnx":  # taille fixée à l'export, lue dans les métadonnées
    import ast
    import onnx
    meta = {p.key: p.value for p in onnx.load(str(args.weights), load_external_data=False).metadata_props}
    IMGSZ = ast.literal_eval(meta.get("imgsz", "[256, 256]"))[0]
else:
    IMGSZ = model.ckpt.get("train_args", {}).get("imgsz", 256) if getattr(model, "ckpt", None) else 256
CAPTURES = ROOT / "captures"
MY_PHOTOS = ROOT / "mes_photos"
COUNTRIES = [NAMES[i] for i in sorted(NAMES)]


ANGLES = np.linspace(0, 2 * np.pi, 90, endpoint=False)


def edge_support(edges, x, y, r):
    """Part du cercle (0-1) qui suit un contour réel, à quelques pixels près (Hough est souvent un peu décalé)."""
    h, w = edges.shape
    hit = np.zeros(len(ANGLES), bool)
    tol = max(2, int(r * 0.06))
    for dr in range(-tol, tol + 1):
        xs = np.clip((x + (r + dr) * np.cos(ANGLES)).astype(int), 0, w - 1)
        ys = np.clip((y + (r + dr) * np.sin(ANGLES)).astype(int), 0, h - 1)
        hit |= edges[ys, xs] > 0
    return float(hit.mean())


def is_real_coin(edges, x, y, r):
    """Le bord doit être net ET le fond juste autour plus calme : sinon c'est une texture prise pour un cercle."""
    return edge_support(edges, x, y, r) >= args.min_edge and edge_support(edges, x, y, int(r * 1.3)) <= args.max_bg


def detect_coins(frame):
    """Retourne les cercles (x, y, r) trouvés dans l'image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=args.min_radius * 2,
        param1=100, param2=args.hough, minRadius=args.min_radius, maxRadius=args.max_radius)
    if circles is None:
        return []
    edges = cv2.Canny(gray, 50, 100)
    h, w = frame.shape[:2]
    kept = []
    for x, y, r in (tuple(int(v) for v in c) for c in circles[0]):  # triés du plus sûr au moins sûr
        if not (0 <= x < w and 0 <= y < h):
            continue
        if min(x, y, w - x, h - y) < r * 0.7:  # pièce trop coupée par le bord de l'image
            continue
        if not is_real_coin(edges, x, y, r):  # faux cercle (ombre, texture, reflet)
            continue
        if any(np.hypot(x - kx, y - ky) < 0.9 * (r + kr) for kx, ky, kr in kept):  # deux pièces ne se superposent pas
            continue
        kept.append((x, y, r))
        if len(kept) == args.max_coins:
            break
    return kept


def crop_coin(frame, x, y, r, mask):
    """Recadrage carré autour de la pièce, avec fond blanc hors du cercle si `mask`."""
    half = int(r * 1.08)
    # padding pour les pièces au bord de l'image
    padded = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    crop = padded[y:y + 2 * half, x:x + 2 * half].copy()
    if mask:
        m = np.zeros(crop.shape[:2], np.uint8)
        cv2.circle(m, (half, half), r, 255, -1)
        crop[m == 0] = 255
    return crop


def classify(crops):
    """Probabilités par pièce (moyennées sur 4 rotations si --tta)."""
    if not crops:
        return []
    n_rot = 4 if args.tta else 1
    batch = [np.ascontiguousarray(np.rot90(c, k)) for c in crops for k in range(n_rot)]
    results = model.predict(batch, imgsz=IMGSZ, verbose=False)
    probs = np.stack([r.probs.data.cpu().numpy() for r in results])
    return probs.reshape(len(crops), n_rot, -1).mean(1)


class Tracker:
    """Associe chaque pièce à celle de l'image précédente la plus proche et lisse ses probabilités."""

    def __init__(self, max_missing=5):
        self.tracks = []  # dicts : x, y, r, probs, missing, hits
        self.max_missing = max_missing

    def update(self, circles, probs):
        """Retourne une piste par cercle, dans le même ordre que `circles`.
        Une piste n'est fiable qu'après `args.confirm` images (voir `confirmed`)."""
        for t in self.tracks:
            t["missing"] += 1
        matched = []
        for (x, y, r), p in zip(circles, probs):
            near = [t for t in self.tracks if t["missing"] > 0 and np.hypot(t["x"] - x, t["y"] - y) < r * 0.6]
            if near:
                t = min(near, key=lambda t: np.hypot(t["x"] - x, t["y"] - y))
                t["probs"] = (1 - args.smooth) * t["probs"] + args.smooth * p
                a = 0.5  # lissage de la position : le cercle affiché ne tremble plus
                t.update(x=int((1 - a) * t["x"] + a * x), y=int((1 - a) * t["y"] + a * y),
                         r=int((1 - a) * t["r"] + a * r), missing=0, hits=t["hits"] + 1)
            else:
                t = {"x": x, "y": y, "r": r, "probs": p, "missing": 0, "hits": 1}
                self.tracks.append(t)
            matched.append(t)
        # une piste non confirmée qui rate une seule image est oubliée : un faux cercle qui clignote
        # n'est jamais affiché ; une pièce confirmée tolère quelques images ratées
        self.tracks = [t for t in self.tracks if t["missing"] <= self.max_missing
                       and (t["hits"] >= args.confirm or t["missing"] == 0)]
        return matched


def confirmed(track):
    return track.get("hits", args.confirm) >= args.confirm


def draw(frame, x, y, r, probs):
    top = np.argsort(probs)[::-1][:3]
    conf = probs[top[0]]
    sure = conf >= args.conf
    color = (0, 200, 0) if sure else (0, 165, 255)
    label = f"{NAMES[top[0]]} {conf:.0%}" if sure else f"incertain ({NAMES[top[0]]}?)"
    cv2.circle(frame, (x, y), r, color, 2)
    cv2.putText(frame, label, (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    for i, k in enumerate(top[1:], 1):  # 2e et 3e choix, en petit
        cv2.putText(frame, f"{NAMES[k]} {probs[k]:.0%}", (x - r, y + r + 20 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def process(frame, tracker, mask, center_mode):
    if center_mode:  # on classe le centre de l'image : poser la pièce dans le cercle guide
        h, w = frame.shape[:2]
        circles = [(w // 2, h // 2, int(min(h, w) * 0.35))]
    else:
        circles = detect_coins(frame)
    crops = [crop_coin(frame, x, y, r, mask) for x, y, r in circles]
    probs = classify(crops)
    tracks = tracker.update(circles, probs) if tracker else [
        {"x": x, "y": y, "r": r, "probs": p} for (x, y, r), p in zip(circles, probs)]
    out = frame.copy()
    # pièces confirmées, y compris celles ratées sur cette image (évite le clignotement)
    shown = [t for t in tracker.tracks if confirmed(t)] if tracker else tracks
    for t in shown:
        draw(out, t["x"], t["y"], t["r"], t["probs"])
    # on ne garde que les pièces confirmées pour la collecte et la sauvegarde
    keep = [i for i, t in enumerate(tracks) if confirmed(t)]
    return out, [crops[i] for i in keep], [tracks[i] for i in keep]


def save_crops(crops, tracks):
    """Sauvegarde les pièces recadrées, rangées par pays prédit : à vérifier puis ajouter au dataset."""
    for crop, t in zip(crops, tracks):
        folder = CAPTURES / NAMES[int(np.argmax(t["probs"]))]
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{time.strftime('%Y%m%d-%H%M%S')}_{t['x']}_{t['y']}.jpg"), crop)
    print(f"{len(crops)} pièce(s) sauvegardée(s) dans {CAPTURES}")


def save_collect(crop, country):
    """Enregistre une photo de pièce dans mes_photos/<Pays>/ et retourne le nombre de photos de ce pays."""
    folder = MY_PHOTOS / country
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"cam_{time.strftime('%Y%m%d-%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"), crop)
    return len(list(folder.glob("*.jpg")))


def count_photos(country):
    return len(list((MY_PHOTOS / country).glob("*.jpg")))


def draw_collect(out, country, n_photos, n_coins, burst, flash):
    """Bandeau du mode collecte : pays choisi, nombre de photos, état de la rafale."""
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, h - 70), (w, h), (40, 40, 40), -1)
    if flash:  # cadre blanc bref à chaque photo
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (255, 255, 255), 8)
    cv2.putText(out, f"< {country} >", (15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
    cv2.putText(out, f"{n_photos} photo(s)", (15, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    if n_coins != 1:
        msg, color = f"{n_coins} piece(s) detectee(s) : en poser exactement 1", (0, 0, 255)
    elif burst:
        msg, color = "RAFALE en cours [r] : tourner la piece", (0, 0, 255)
    else:
        msg, color = "Espace : photo | r : rafale | fleches : pays", (0, 255, 0)
    cv2.putText(out, msg, (w // 2 - 80, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


if args.image:
    frame = cv2.imread(str(args.image))
    scale = 1280 / max(frame.shape[:2])
    if scale < 1:  # même résolution qu'une webcam : la détection de cercles reste rapide
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out, _, tracks = process(frame, None, mask=True, center_mode=False)
    if not tracks:  # aucune pièce détectée : on suppose que la photo est déjà cadrée sur la pièce
        out, _, tracks = process(frame, None, mask=True, center_mode=True)
    for t in tracks:
        k = int(np.argmax(t["probs"]))
        print(f"pièce ({t['x']}, {t['y']}) -> {NAMES[k]} {t['probs'][k]:.1%}")
    cv2.imshow("2 euros", out)
    cv2.waitKey(0)
    raise SystemExit

cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise SystemExit(f"Impossible d'ouvrir la webcam {args.camera}")

tracker = Tracker()
mask, center_mode = True, False
collect, burst, country_idx = args.collect, False, 0
n_photos, last_shot, flash_until = count_photos(COUNTRIES[0]), 0.0, 0.0
LEFT, RIGHT = {2424832, 65361, ord("a")}, {2555904, 65363, ord("d")}  # flèches (Windows, Linux) et a/d
fps, t0 = 0.0, time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    out, crops, tracks = process(frame, tracker, mask, center_mode)

    if center_mode and not tracks:
        h, w = frame.shape[:2]
        cv2.circle(out, (w // 2, h // 2), int(min(h, w) * 0.35), (255, 255, 255), 1)
    fps = 0.9 * fps + 0.1 / max(time.time() - t0, 1e-6)
    t0 = time.time()
    if collect:
        now = time.time()
        if burst and len(crops) == 1 and now - last_shot >= args.burst:
            n_photos, last_shot, flash_until = save_collect(crops[0], COUNTRIES[country_idx]), now, now + 0.1
        draw_collect(out, COUNTRIES[country_idx], n_photos, len(crops), burst, now < flash_until)
    status = f"{fps:4.1f} fps | masque {'on' if mask else 'off'} [m] | mode {'centre' if center_mode else 'detection'} [c] | collecte [k] | sauver [s] | quitter [q]"
    cv2.putText(out, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(out, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("2 euros", out)

    key = cv2.waitKeyEx(1)
    if key in LEFT | RIGHT and collect:
        country_idx = (country_idx + (1 if key in RIGHT else -1)) % len(COUNTRIES)
        n_photos, burst = count_photos(COUNTRIES[country_idx]), False
    key = key & 0xFF if key != -1 else -1
    if key == ord("k"):
        collect, burst = not collect, False
    if collect and key == ord("r"):
        burst = not burst
    if collect and key == ord(" ") and len(crops) == 1:
        n_photos, flash_until = save_collect(crops[0], COUNTRIES[country_idx]), time.time() + 0.1
    if key == ord("q"):
        break
    if key == ord("m"):
        mask = not mask
    if key == ord("c"):
        center_mode = not center_mode
        tracker = Tracker()
    if key == ord("s") and crops:
        save_crops(crops, tracks)

cap.release()
cv2.destroyAllWindows()
