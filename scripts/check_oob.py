from pathlib import Path
lbl = Path(__file__).resolve().parents[1] / "datasets" / "project-497" / "labels"
oob_files = oob_boxes = tot = 0
maxov = 0.0
for f in lbl.glob("*.txt"):
    bad = False
    for ln in f.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        _, cx, cy, w, h = ln.split()
        cx, cy, w, h = map(float, (cx, cy, w, h))
        tot += 1
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
            oob_boxes += 1
            bad = True
            maxov = max(maxov, x2 - 1, y2 - 1, -x1, -y1)
    if bad:
        oob_files += 1
print(f"files_with_oob={oob_files} oob_boxes={oob_boxes}/{tot} ({100*oob_boxes/tot:.2f}%) max_overflow={maxov:.3f}")
