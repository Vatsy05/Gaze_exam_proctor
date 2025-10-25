# infer_realtime.py  — stable build (no head pose)
import os, time, cv2, numpy as np
from collections import deque, Counter
from datetime import datetime
import mediapipe as mp
from tensorflow.keras.models import load_model

# ----------- Config -----------
MODEL_PATH = "gaze_cnn.h5"
LABELS_FILE = "labels.txt"
EYE_SIZE = (64, 64)

# Smoothing & thresholds
SMOOTH_WINDOW = 25            # long majority vote fallback
AWAY_RATIO_WINDOW = 15        # window for away ratio
AWAY_RATIO_THRESH = 0.6       # >=60% of last N frames => away
CLOSED_RECENT_WINDOW = 5      # look at last 5 frames for closed
CLOSED_RECENT_MIN = 2         # if >=2 say closed, force closed
CLOSED_FORCE_CONF = 0.60      # or if closed prob >= 0.60, force closed

THRESH_SECONDS = 2.5          # sustained away duration
CLOSED_TIMEOUT = 5.0          # eyes closed duration (seconds)
VIDEO_BUFFER_SECONDS = 5.0
FPS_ASSUMED = 20
FLASH_DURATION = 15           # frames (~0.5s at 30 fps)

LOG_DIR = "events"
os.makedirs(LOG_DIR, exist_ok=True)

# ----------- Labels -----------
with open(LABELS_FILE) as f:
    LABELS = [l.strip() for l in f if l.strip()]

LOOKING_CENTER = "center"

# ----------- MediaPipe -----------
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# We track ONE main face for gaze; use FaceDetection just to count faces.
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)

# Eye landmarks
LEFT  = [33,133,160,159,158,153,144]
RIGHT = [362,263,387,386,385,380,373]

# ----------- Helpers -----------
def eye_bbox(lm, w, h, idxs, pad=8):
    xs = [int(lm[i].x * w) for i in idxs]
    ys = [int(lm[i].y * h) for i in idxs]
    x1, x2 = max(min(xs)-pad,0), min(max(xs)+pad, w-1)
    y1, y2 = max(min(ys)-pad,0), min(max(ys)+pad, h-1)
    return x1,y1,x2,y2

def crop_eyes(frame, lm):
    h, w = frame.shape[:2]
    lx1,ly1,lx2,ly2 = eye_bbox(lm, w, h, LEFT)
    rx1,ry1,rx2,ry2 = eye_bbox(lm, w, h, RIGHT)
    L = frame[ly1:ly2, lx1:lx2]
    R = frame[ry1:ry2, rx1:rx2]
    if L.size == 0 or R.size == 0:
        return None
    L = cv2.resize(L, EYE_SIZE); R = cv2.resize(R, EYE_SIZE)
    return L, R

def preprocess_pair(L, R):
    L = cv2.cvtColor(L, cv2.COLOR_BGR2RGB).astype("float32")/255.0
    R = cv2.cvtColor(R, cv2.COLOR_BGR2RGB).astype("float32")/255.0
    X = np.concatenate([L, R], axis=-1)  # (64,64,6)
    return np.expand_dims(X, axis=0)     # (1,64,64,6)

def save_evidence(frames, label, fps_est):
    if not frames:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # macOS-friendly
    out = cv2.VideoWriter(os.path.join(LOG_DIR, f"evidence_{label}_{ts}.mp4"),
                          fourcc, fps_est, (w,h))
    for f in frames:
        out.write(f)
    out.release()
    with open(os.path.join(LOG_DIR, "events.log"), "a") as f:
        f.write(f"{ts},{label}\n")
    print(f"[EVENT] {ts} - {label} (evidence saved)")

def add_red_flash(frame, text="WARNING!"):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0,0), (w,h), (0,0,255), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    cv2.putText(frame, text, (50,150),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 5)
    return frame

# ----------- Main loop -----------
def main():
    model = load_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = FPS_ASSUMED
    buf_len = int(fps * VIDEO_BUFFER_SECONDS)

    # Buffers
    long_buf   = deque(maxlen=SMOOTH_WINDOW)          # raw labels for majority fallback
    away_buf   = deque(maxlen=AWAY_RATIO_WINDOW)      # booleans "away?"
    closed_buf = deque(maxlen=CLOSED_RECENT_WINDOW)   # booleans "closed?"

    video_buf = deque(maxlen=buf_len)

    away_start = None
    closed_start = None
    event_armed = True

    flash_counter = 0
    flash_text = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- Face count with FaceDetection ----
        res_det = face_detection.process(rgb)
        face_num = len(res_det.detections) if res_det and res_det.detections else 0
        cv2.putText(frame, f"faces: {face_num}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,180), 2)

        if face_num > 1:
            flash_counter = FLASH_DURATION
            flash_text = "MULTIPLE FACES DETECTED!"
            save_evidence(list(video_buf), "multiple_faces", fps)

        # ---- FaceMesh for gaze on main face ----
        res_mesh = face_mesh.process(rgb)
        faces_mesh = res_mesh.multi_face_landmarks if res_mesh and res_mesh.multi_face_landmarks else []

        label_disp = "no_face"
        conf_disp = 0.0

        if len(faces_mesh) == 1:
            lm = faces_mesh[0].landmark
            eyes = crop_eyes(frame, lm)
            if eyes:
                L, R = eyes
                X = preprocess_pair(L, R)
                probs = model.predict(X, verbose=0)[0]
                idx = int(np.argmax(probs))
                raw_label = LABELS[idx]
                conf = float(probs[idx])

                # Update buffers
                long_buf.append(raw_label)
                p_closed = probs[LABELS.index("closed")] if "closed" in LABELS else 0.0
                is_closed = (raw_label == "closed") or (p_closed >= CLOSED_FORCE_CONF)
                is_away = (raw_label not in [LOOKING_CENTER, "closed"])
                closed_buf.append(bool(is_closed))
                away_buf.append(bool(is_away))

                # -------- Priority decision --------
                # 1) Closed wins quickly
                if sum(closed_buf) >= CLOSED_RECENT_MIN or is_closed:
                    disp_class = "closed"

                # 2) Otherwise, sustained away ratio
                elif len(away_buf) > 0 and (sum(away_buf)/len(away_buf) >= AWAY_RATIO_THRESH):
                    disp_class = "away"

                # 3) Otherwise, long majority smoothing
                else:
                    if len(long_buf) > 0:
                        counts = Counter(long_buf)
                        label_sm = max(counts, key=counts.get)
                    else:
                        label_sm = raw_label
                    if label_sm == LOOKING_CENTER:
                        disp_class = "center"
                    elif label_sm == "closed":
                        disp_class = "closed"
                    else:
                        disp_class = "away"

                label_disp, conf_disp = disp_class, conf

                # Eye thumbnails (top-left)
                hE, wE = L.shape[:2]
                frame[5:5+hE, 5:5+wE] = L
                frame[5:5+hE, 10+wE:10+2*wE] = R

                # --- Sustained away event ---
                if disp_class == "away":
                    if away_start is None:
                        away_start = time.time()
                    else:
                        if (time.time() - away_start) >= THRESH_SECONDS and event_armed:
                            flash_counter = FLASH_DURATION
                            flash_text = "LOOKING AWAY!"
                            save_evidence(list(video_buf), "away", fps)
                            event_armed = False
                else:
                    away_start = None
                    event_armed = True

                # --- Closed-eyes timeout event ---
                if disp_class == "closed":
                    if closed_start is None:
                        closed_start = time.time()
                    else:
                        if (time.time() - closed_start) >= CLOSED_TIMEOUT:
                            flash_counter = FLASH_DURATION
                            flash_text = "EYES CLOSED TOO LONG!"
                            save_evidence(list(video_buf), "eyes_closed_too_long", fps)
                            closed_start = None  # reset after firing
                else:
                    closed_start = None

            else:
                long_buf.clear(); away_buf.clear(); closed_buf.clear()
                away_start = None; closed_start = None; event_armed = True
        else:
            long_buf.clear(); away_buf.clear(); closed_buf.clear()
            away_start = None; closed_start = None; event_armed = True

        # ---- Overlay label ----
        cv2.putText(frame, f"{label_disp} ({conf_disp:.2f})", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0) if label_disp=="center" else (0,255,255), 2)

        # ---- Red flash ----
        if flash_counter > 0:
            frame = add_red_flash(frame, flash_text)
            flash_counter -= 1

        # Keep evidence buffer & show
        video_buf.append(frame.copy())
        cv2.imshow("Exam Gaze Proctor — press q to quit", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
