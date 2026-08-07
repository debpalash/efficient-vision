from collections import Counter
from pathlib import Path
from PIL import Image

DS = Path(__file__).resolve().parents[1] / "datasets" / "project-497" / "yolo"
for split in ("train", "val"):
    c = Counter()
    for p in (DS / "images" / split).iterdir():
        if p.is_file():
            with Image.open(p) as im:
                c[f"{im.size[0]}x{im.size[1]}"] += 1
    print(f"--- {split} ({sum(c.values())} images, {len(c)} distinct sizes) ---")
    for k, v in c.most_common(14):
        print(f"  {k:>10} {v:>5}")
