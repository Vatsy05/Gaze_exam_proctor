# scripts/import_columbia.py
import os, re, shutil
from pathlib import Path

# Your layout: Exam_Proctor/open_gaze/Columbia Gaze Data Set/0001/*.jpg
IMAGES_DIR = "../open_gaze/Columbia Gaze Data Set"
TARGET_DIR = "../raw_images"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Center tolerance in degrees (dataset uses {-15,-10,-5,0,5,10,15})
CENTER_T = 5

# Regex to capture "..._<P>P_<V>V_<H>H.jpg" with optional signs
# Examples: -15P_-10V_-5H , 0P_0V_0H , 10P_15V_-5H
PAT = re.compile(r"([+-]?\d+)P_([+-]?\d+)V_([+-]?\d+)H", re.IGNORECASE)

def label_from_angles(H, V, center_t=CENTER_T):
    # H: left/right (yaw-ish), V: up/down (pitch-ish)
    if abs(H) <= center_t and abs(V) <= center_t:
        return "center"
    if H < -center_t:  # negative H = look left
        return "left"
    if H > center_t:   # positive H = look right
        return "right"
    if V < -center_t:  # negative V = look up
        return "up"
    if V > center_t:   # positive V = look down
        return "down"
    return None

def safe_copy(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    i = 1
    while dst.exists():
        dst = dst_dir / f"{src.stem}_{i}{src.suffix}"
        i += 1
    shutil.copy2(src, dst)

def main():
    root = Path(IMAGES_DIR)
    copied = skipped = 0
    for subj in sorted(root.iterdir()):
        if not subj.is_dir():
            continue
        for img in subj.iterdir():
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue
            m = PAT.search(img.stem)
            if not m:
                skipped += 1
                continue
            try:
                P = int(m.group(1))   # not used for label, but available
                V = int(m.group(2))
                H = int(m.group(3))
            except Exception:
                skipped += 1
                continue

            lbl = label_from_angles(H, V)
            if not lbl:
                skipped += 1
                continue

            safe_copy(img, Path(TARGET_DIR) / lbl)
            copied += 1

    print(f"[done] copied={copied}, skipped={skipped}")
    print("Merged into raw_images/<center|left|right|up|down>/ (keep 'closed' from your webcam).")

if __name__ == "__main__":
    main()
