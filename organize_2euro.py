"""Range les images 2€ du EURO-Coin Dataset dans un dossier par pays.

raw/Dataset_Value/2-euro/2-euro-Germany_42.jpg -> 2euro_by_country/Germany/2-euro-Germany_42.jpg
Format compatible avec torchvision.datasets.ImageFolder / Keras image_dataset_from_directory.
"""
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "raw" / "Dataset_Value" / "2-euro"
DST = ROOT / "2euro_by_country"

pattern = re.compile(r"^2-euro-(.+)_\d+\.(jpe?g|png|gif)$", re.IGNORECASE)
# Certains noms de pays sont en italien dans le dataset d'origine
ITALIAN_TO_EN = {
    "Belgio": "Belgium", "Cipro": "Cyprus", "Finlandia": "Finland",
    "Lettonia": "Latvia", "Lituania": "Lithuania", "Paesi-Bassi": "Netherlands",
}
if DST.exists():
    shutil.rmtree(DST)
counts = Counter()
for img in SRC.iterdir():
    m = pattern.match(img.name)
    if not m:
        print("ignoré:", img.name)
        continue
    country = ITALIAN_TO_EN.get(m.group(1), m.group(1))
    (DST / country).mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, DST / country / img.name)
    counts[country] += 1

for country, n in sorted(counts.items()):
    print(f"{country:15} {n}")
print("total", sum(counts.values()))
