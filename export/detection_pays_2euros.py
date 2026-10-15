"""Détection en direct du pays d'une pièce de 2 € avec une webcam (modèle modeles/pays_2euros.onnx).

Le modèle reconnaît la face nationale de 23 pays (Allemagne, Belgique, France...). Il suppose qu'on
lui montre une pièce de 2 € : sur autre chose, il répond quand même un pays. Pour vérifier d'abord
que c'est bien un euro, utiliser detection_euro.py.

Pipeline :
1. Détection des cercles (Hough), puis vérification que le bord est un vrai contour : élimine
   les faux cercles dus aux ombres, reflets et textures.
2. Recadrage carré sur la pièce, fond blanc hors du cercle (comme les images d'entraînement).
3. Classification par le modèle ONNX.
4. Suivi d'une image à l'autre : une pièce n'est affichée qu'après plusieurs images consécutives,
   et ses probabilités sont lissées.
5. Seuil de confiance : en dessous, la pièce est affichée comme "incertain".

Dépendances : opencv-python, numpy, onnxruntime, pillow (pas besoin d'ultralytics ni de PyTorch).

Usage :
    python detection_pays_2euros.py                  # webcam 0
    python detection_pays_2euros.py --camera 1
    python detection_pays_2euros.py --image photo.jpg

Depuis un autre programme :
    from detection_pays_2euros import Pays2EurosDetector
    detector = Pays2EurosDetector()
    for coin in detector.detect(frame):     # frame BGR (OpenCV)
        print(coin.label, coin.conf, coin.x, coin.y, coin.r)

Touches : q quitter | m masque on/off
"""
import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).parent
DEFAULT_MODEL = HERE / "modeles" / "pays_2euros.onnx"
ANGLES = np.linspace(0, 2 * np.pi, 90, endpoint=False)
BIMETAL = 1.39  # rayon de l'anneau / rayon du cœur d'une pièce de 2 € (1,37) ou de 1 € (1,43)


@dataclass
class Settings:
    conf: float = 0.6         # confiance minimale pour afficher une classe
    min_radius: int = 40      # rayon minimal d'une pièce (px)
    max_radius: int = 250     # rayon maximal d'une pièce (px)
    max_coins: int = 5        # nombre maximal de pièces par image
    hough: int = 60           # seuil de Hough : plus haut = moins de faux cercles
    min_edge: float = 0.6     # part minimale du cercle qui doit suivre un vrai contour (0-1)
    max_bg: float = 0.5       # part maximale de contours juste autour de la pièce (0-1)
    confirm: int = 4          # images consécutives avant d'afficher une pièce
    smooth: float = 0.3       # poids de la nouvelle image dans la moyenne des probabilités (0-1)
    mask: bool = True         # fond blanc hors du cercle


@dataclass
class Coin:
    x: int
    y: int
    r: int
    label: str                # classe la plus probable
    conf: float               # sa probabilité
    probs: np.ndarray         # probabilités de toutes les classes
    sure: bool                # conf >= Settings.conf


class OnnxClassifier:
    """Modèle de classification YOLO exporté en ONNX : même prétraitement qu'ultralytics."""

    def __init__(self, path):
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = ast.literal_eval(meta["names"])
        self.imgsz = ast.literal_eval(meta.get("imgsz", "[256, 256]"))[0]
        self.input = self.session.get_inputs()[0].name

    def __call__(self, crop):
        """Probabilités des classes pour une image BGR carrée."""
        # redimensionnement par PIL : résultat identique à ultralytics (cv2.resize s'en écarte)
        img = np.asarray(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize(
            (self.imgsz, self.imgsz), Image.BILINEAR))
        img = img.astype(np.float32) / 255.0
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None])
        return self.session.run(None, {self.input: x})[0][0]  # déjà passées par softmax


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


def detect_circles(frame, s):
    """Cercles (x, y, r) qui ressemblent à de vraies pièces."""
    gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=s.min_radius * 2,
        param1=100, param2=s.hough, minRadius=s.min_radius, maxRadius=s.max_radius)
    if circles is None:
        return []
    edges = cv2.Canny(gray, 50, 100)
    h, w = frame.shape[:2]
    kept = []
    for x, y, r in (tuple(int(v) for v in c) for c in circles[0]):  # triés du plus sûr au moins sûr
        if not (0 <= x < w and 0 <= y < h):
            continue
        if edge_support(edges, x, y, r) < s.min_edge:  # faux cercle (ombre, reflet)
            continue
        if edge_support(edges, x, y, int(r * BIMETAL)) >= 0.5:  # c'était le cœur d'une pièce bimétallique
            r = int(r * BIMETAL)
        if edge_support(edges, x, y, int(r * 1.3)) > s.max_bg:  # fond trop chargé autour : une texture
            continue
        if min(x, y, w - x, h - y) < r * 0.7:  # pièce trop coupée par le bord de l'image
            continue
        if any(np.hypot(x - kx, y - ky) < 0.9 * (r + kr) for kx, ky, kr in kept):  # pas de superposition
            continue
        kept.append((x, y, r))
        if len(kept) == s.max_coins:
            break
    return kept


def crop_coin(frame, x, y, r, mask):
    """Recadrage carré autour de la pièce, avec fond blanc hors du cercle si `mask`."""
    half = int(r * 1.08)
    padded = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    crop = padded[y:y + 2 * half, x:x + 2 * half].copy()
    if mask:
        m = np.zeros(crop.shape[:2], np.uint8)
        cv2.circle(m, (half, half), r, 255, -1)
        crop[m == 0] = 255
    return crop


class Tracker:
    """Suit chaque pièce d'une image à l'autre, lisse sa position et ses probabilités."""

    def __init__(self, s, max_missing=5):
        self.s = s
        self.max_missing = max_missing
        self.tracks = []  # dicts : x, y, r, probs, missing, hits

    def update(self, circles, probs):
        for t in self.tracks:
            t["missing"] += 1
        for (x, y, r), p in zip(circles, probs):
            near = [t for t in self.tracks if t["missing"] > 0 and np.hypot(t["x"] - x, t["y"] - y) < r * 0.6]
            if near:
                t = min(near, key=lambda t: np.hypot(t["x"] - x, t["y"] - y))
                t["probs"] = (1 - self.s.smooth) * t["probs"] + self.s.smooth * p
                t.update(x=(t["x"] + x) // 2, y=(t["y"] + y) // 2, r=(t["r"] + r) // 2,
                         missing=0, hits=t["hits"] + 1)
            else:
                self.tracks.append({"x": x, "y": y, "r": r, "probs": p, "missing": 0, "hits": 1})
        # une piste non confirmée qui rate une image est oubliée (faux cercle qui clignote) ;
        # une pièce confirmée tolère quelques images ratées
        self.tracks = [t for t in self.tracks if t["missing"] <= self.max_missing
                       and (t["hits"] >= self.s.confirm or t["missing"] == 0)]
        return [t for t in self.tracks if t["hits"] >= self.s.confirm]


class Pays2EurosDetector:
    """Détecte les pièces de 2 € dans une image et donne leur pays."""

    def __init__(self, model=DEFAULT_MODEL, settings=None, track=True):
        self.s = settings or Settings()
        self.model = OnnxClassifier(model)
        self.tracker = Tracker(self.s) if track else None  # track=False pour des photos indépendantes

    def detect(self, frame):
        """Liste des pièces confirmées dans l'image BGR `frame`."""
        circles = detect_circles(frame, self.s)
        probs = [self.model(crop_coin(frame, x, y, r, self.s.mask)) for x, y, r in circles]
        if self.tracker:
            tracks = self.tracker.update(circles, probs)
        else:
            tracks = [{"x": x, "y": y, "r": r, "probs": p} for (x, y, r), p in zip(circles, probs)]
        coins = []
        for t in tracks:
            k = int(np.argmax(t["probs"]))
            conf = float(t["probs"][k])
            coins.append(Coin(t["x"], t["y"], t["r"], self.model.names[k], conf, t["probs"], conf >= self.s.conf))
        return coins


def draw(frame, coins, names):
    for c in coins:
        color = (0, 200, 0) if c.sure else (0, 165, 255)
        text = f"{c.label} {c.conf:.0%}" if c.sure else f"incertain ({c.label}?)"
        cv2.circle(frame, (c.x, c.y), c.r, color, 2)
        cv2.putText(frame, text, (c.x - c.r, c.y - c.r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        for i, k in enumerate(np.argsort(c.probs)[::-1][1:3], 1):  # 2e et 3e choix, en petit
            cv2.putText(frame, f"{names[k]} {c.probs[k]:.0%}", (c.x - c.r, c.y + c.r + 20 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    p.add_argument("--camera", default=0, type=int, help="index de la webcam")
    p.add_argument("--image", type=Path, help="analyser une photo au lieu de la webcam")
    for name, default in vars(Settings()).items():
        if not isinstance(default, bool):
            p.add_argument(f"--{name.replace('_', '-')}", default=default, type=type(default))
    args = p.parse_args()
    s = Settings(**{k: getattr(args, k) for k in vars(Settings()) if hasattr(args, k)})

    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"Image illisible : {args.image}")
        scale = 1280 / max(frame.shape[:2])
        if scale < 1:  # même résolution qu'une webcam
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        detector = Pays2EurosDetector(args.model, s, track=False)
        names = detector.model.names
        coins = detector.detect(frame)
        for c in coins:
            print(f"({c.x}, {c.y}) -> {c.label} {c.conf:.1%}")
        if not coins:
            print("Aucune pièce détectée")
        cv2.imshow("Pays 2 euros", draw(frame, coins, names))
        cv2.waitKey(0)
        return

    detector = Pays2EurosDetector(args.model, s)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir la webcam {args.camera}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = draw(frame.copy(), detector.detect(frame), detector.model.names)
        status = f"masque {'on' if s.mask else 'off'} [m] | quitter [q]"
        cv2.putText(out, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(out, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Pays 2 euros", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            s.mask = not s.mask
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
