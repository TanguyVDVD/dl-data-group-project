"""Transformations d'entraînement partagées par les notebooks.

Elles sont dans un fichier à part (et pas dans le notebook) : sous Windows, les processus qui chargent
les images (workers) doivent pouvoir importer ces classes.
"""
import random

from PIL import Image, ImageDraw


class CircleMask:
    """Avec probabilité p, met en blanc tout ce qui est hors du cercle inscrit (comme le masque de camera.py)."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).ellipse((0, 0, img.width - 1, img.height - 1), fill=255)
        return Image.composite(img, Image.new(img.mode, img.size, "white"), mask)

    def __repr__(self):
        return f"CircleMask(p={self.p})"
