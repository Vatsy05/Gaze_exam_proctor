# preprocess.py
import os
import csv
import cv2
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp
from collections import Counter

# ----- paths & config -----
INPUT_DIR = "raw_images"     # expects subfolders: center/left/right/up/down/closed
OUT_DIR   = "processed"      # eye crops will be saved here by label
CSV_OUT   = "data.csv"       # merged dataset manifest
EYE_SIZE  = (64, 64)         # crop size (H, W)
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# MediaPipe Face Mesh (static mode = per-image)
mp_face_mesh = mp.solutions.face_mesh

# Landmark indices around eyes (MediaPipe)
LEFT  = [33,133,160,159,158,153,144]
RIGHT = [362,263,387,386,385,380,373]

def eye_bbox(lm, w, h, idxs, pad=8):
    xs = [int(lm[i].x * w) for i in idxs]
    ys = [int(lm[i].y * h) for i in idxs]
    x1, x2 = max(min(xs) - pad, 0), min(max(xs) + pad, w - 1)
    y1, y2 = max(min(ys) - pad, 0), min(max(ys) + pad, h - 1)
    return x1, y1, x2, y2

def extract_eyes(img_bgr):
    """Return (left_eye_bgr, right_eye_bgr) resized to EYE_SIZE, or None."""
    h, w = img_bgr.shape[:2]
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True) as fm:
        res = fm.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        lx1, ly1, lx2, ly2 = eye_bbox(lm, w, h, LEFT)
        rx1, ry1, rx2, ry2 = eye_bbox(lm, w, h, RIGHT)
        L = img_bgr[ly1:ly2, lx1:lx2]
        R = img_bgr[ry1:ry2, rx1:rx2]
        if L.size == 0 or R.size == 0:
            return None
        L = cv2.resize(L, EYE_SIZE, interpolation=cv2.INTER_AREA)
        R = cv2.resize(R, EYE_SIZE, interpolation=cv2.INTER_AREA)
        return L, R

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    rows = []
    per_label_counts = Counter()
    skipped_no_face = 0
    skipped_bad = 0

    # iterate label folders
    labels = [d for d in sorted(os.listdir(INPUT_DIR)) if (Path(INPUT_DIR)/d).is_dir()]
    if not labels:
        print(f"[ERR] No label folders found in {INPUT_DIR}/")
        return

    for label in labels:
        in_dir  = Path(INPUT_DIR) / label
        out_dir = Path(OUT_DIR) / label
        out_dir.mkdir(parents=True, exist_ok=True)

        files = [p for p in in_dir.iterdir() if is_image_file(p)]
        if not files:
            print(f"[WARN] No images in {in_dir}")
            continue

        for img_path in tqdm(files, desc=label):
            img = cv2.imread(str(img_path))
            if img is None:
                skipped_bad += 1
                continue
            eyes = extract_eyes(img)
            if eyes is None:
                skipped_no_face += 1
                continue

            L, R = eyes
            stem = img_path.stem
            lp = out_dir / f"{stem}_L.jpg"
            rp = out_dir / f"{stem}_R.jpg"
            cv2.imwrite(str(lp), L)
            cv2.imwrite(str(rp), R)
            rows.append([str(lp), str(rp), label])
            per_label_counts[label] += 1  # counts pairs (one row per pair)

    # write CSV
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["left", "right", "label"])
        w.writerows(rows)

    # summary
    print(f"\n[OK] processed {len(rows)} samples (pairs) -> {CSV_OUT}")
    print("[per-class processed counts]")
    for k in sorted(per_label_counts.keys()):
        print(f"  {k}: {per_label_counts[k]}")
    print(f"[skipped] no_face_or_landmarks: {skipped_no_face}, unreadable_files: {skipped_bad}")
    print(f"[out] eye crops written under: {OUT_DIR}/<label>/")

if __name__ == "__main__":
    main()
