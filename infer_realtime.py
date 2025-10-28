import os, time, cv2, numpy as np, wave, threading
from collections import deque, Counter
from datetime import datetime
import mediapipe as mp
from tensorflow.keras.models import load_model
import sounddevice as sd

MODEL_PATH = "gaze_cnn.h5"
LABELS_FILE = "labels.txt"
EYE_SIZE = (64, 64)

SMOOTH_WINDOW = 25
AWAY_RATIO_WINDOW = 15
AWAY_RATIO_THRESH = 0.6
CLOSED_RECENT_WINDOW = 5
CLOSED_RECENT_MIN = 2
CLOSED_FORCE_CONF = 0.60

THRESH_SECONDS = 2.5
CLOSED_TIMEOUT = 5.0
VIDEO_BUFFER_SECONDS = 5.0
FPS_ASSUMED = 20
FLASH_DURATION = 15

AUDIO_SR = 16000
AUDIO_BLOCK = 1024
AUDIO_BUFFER_SECONDS = 6.0
VOICE_MIN_SECONDS = 1.0
VOICE_COOLDOWN = 6.0
VOICE_THRESH_RMS = 0.03
VOICE_SMOOTH = 8

LOG_DIR = "events"
os.makedirs(LOG_DIR, exist_ok=True)

with open(LABELS_FILE) as f:
    LABELS = [l.strip() for l in f if l.strip()]
LOOKING_CENTER = "center"

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

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

LEFT  = [33,133,160,159,158,153,144]
RIGHT = [362,263,387,386,385,380,373]

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
    X = np.concatenate([L, R], axis=-1)
    return np.expand_dims(X, axis=0)

def save_evidence_video(frames, label, fps_est):
    if not frames:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(os.path.join(LOG_DIR, f"evidence_{label}_{ts}.mp4"),
                          fourcc, fps_est, (w,h))
    for f in frames:
        out.write(f)
    out.release()
    with open(os.path.join(LOG_DIR, "events.log"), "a") as f:
        f.write(f"{ts},{label}\n")
    print(f"[EVENT] {ts} - {label} (video saved)")

def save_evidence_audio(samples_int16, label):
    if samples_int16 is None or len(samples_int16) == 0:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(LOG_DIR, f"evidence_{label}_{ts}.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_SR)
        wf.writeframes(samples_int16.tobytes())
    with open(os.path.join(LOG_DIR, "events.log"), "a") as f:
        f.write(f"{ts},{label}\n")
    print(f"[EVENT] {ts} - {label} (audio saved)")

def add_red_flash(frame, text="WARNING!"):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0,0), (w,h), (0,0,255), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    cv2.putText(frame, text, (50,150),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 5)
    return frame

class AudioMonitor:
    def __init__(self):
        self.buffer = deque(maxlen=int(AUDIO_SR * AUDIO_BUFFER_SECONDS))
        self.rms_buf = deque(maxlen=VOICE_SMOOTH)
        self.lock = threading.Lock()
        self._speaking = False
        self._speak_start = None
        self._last_trigger = 0.0
        self.just_triggered = False
        self.running = False
        self.stream = None

    @staticmethod
    def _rms_int16(x):
        if x.size == 0:
            return 0.0
        y = x.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(y*y)))

    def _callback(self, indata, frames, time_info, status):
        samples = indata.copy().ravel().astype(np.int16)
        with self.lock:
            self.buffer.extend(samples.tolist())
        rms = self._rms_int16(samples)
        self.rms_buf.append(rms)
        rms_smooth = sum(self.rms_buf)/len(self.rms_buf)
        now = time.time()
        if rms_smooth >= VOICE_THRESH_RMS:
            if not self._speaking:
                self._speaking = True
                self._speak_start = now
            else:
                dur = now - (self._speak_start or now)
                if dur >= VOICE_MIN_SECONDS and (now - self._last_trigger) >= VOICE_COOLDOWN:
                    self.just_triggered = True
                    self._last_trigger = now
        else:
            self._speaking = False
            self._speak_start = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.stream = sd.InputStream(
            channels=1,
            samplerate=AUDIO_SR,
            dtype="int16",
            blocksize=AUDIO_BLOCK,
            callback=self._callback
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.running = False

    def get_rms(self):
        if not self.rms_buf:
            return 0.0
        return sum(self.rms_buf)/len(self.rms_buf)

    def grab_recent_audio(self, seconds=5.0):
        with self.lock:
            n = int(AUDIO_SR * seconds)
            if len(self.buffer) < 10:
                return None
            data = np.array(list(self.buffer)[-n:], dtype=np.int16)
        return data

def main():
    model = load_model(MODEL_PATH)
    audio_mon = AudioMonitor()
    try:
        audio_mon.start()
    except Exception as e:
        print("[Audio] Could not start microphone stream:", e)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        audio_mon.stop()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = FPS_ASSUMED
    buf_len = int(fps * VIDEO_BUFFER_SECONDS)

    long_buf   = deque(maxlen=SMOOTH_WINDOW)
    away_buf   = deque(maxlen=AWAY_RATIO_WINDOW)
    closed_buf = deque(maxlen=CLOSED_RECENT_WINDOW)
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

        res_det = face_detection.process(rgb)
        face_num = len(res_det.detections) if res_det and res_det.detections else 0
        cv2.putText(frame, f"faces: {face_num}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,180), 2)

        if face_num > 1:
            flash_counter = FLASH_DURATION
            flash_text = "MULTIPLE FACES DETECTED!"
            save_evidence_video(list(video_buf), "multiple_faces", fps)

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

                long_buf.append(raw_label)
                p_closed = probs[LABELS.index("closed")] if "closed" in LABELS else 0.0
                is_closed = (raw_label == "closed") or (p_closed >= CLOSED_FORCE_CONF)
                is_away = (raw_label not in [LOOKING_CENTER, "closed"])
                closed_buf.append(bool(is_closed))
                away_buf.append(bool(is_away))

                if sum(closed_buf) >= CLOSED_RECENT_MIN or is_closed:
                    disp_class = "closed"
                elif len(away_buf) > 0 and (sum(away_buf)/len(away_buf) >= AWAY_RATIO_THRESH):
                    disp_class = "away"
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

                hE, wE = L.shape[:2]
                frame[5:5+hE, 5:5+wE] = L
                frame[5:5+hE, 10+wE:10+2*wE] = R

                if disp_class == "away":
                    if away_start is None:
                        away_start = time.time()
                    else:
                        if (time.time() - away_start) >= THRESH_SECONDS and event_armed:
                            flash_counter = FLASH_DURATION
                            flash_text = "LOOKING AWAY!"
                            save_evidence_video(list(video_buf), "away", fps)
                            event_armed = False
                else:
                    away_start = None
                    event_armed = True

                if disp_class == "closed":
                    if closed_start is None:
                        closed_start = time.time()
                    else:
                        if (time.time() - closed_start) >= CLOSED_TIMEOUT:
                            flash_counter = FLASH_DURATION
                            flash_text = "EYES CLOSED TOO LONG!"
                            save_evidence_video(list(video_buf), "eyes_closed_too_long", fps)
                            closed_start = None
                else:
                    closed_start = None
            else:
                long_buf.clear(); away_buf.clear(); closed_buf.clear()
                away_start = None; closed_start = None; event_armed = True
        else:
            long_buf.clear(); away_buf.clear(); closed_buf.clear()
            away_start = None; closed_start = None; event_armed = True

        rms = audio_mon.get_rms()
        rms_bar = min(int(rms * 200), 200)
        cv2.rectangle(frame, (10, frame.shape[0]-20), (10 + rms_bar, frame.shape[0]-10), (255,255,0), -1)
        cv2.putText(frame, "audio", (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        if getattr(audio_mon, "just_triggered", False):
            audio_mon.just_triggered = False
            flash_counter = FLASH_DURATION
            flash_text = "VOICE DETECTED!"
            recent = audio_mon.grab_recent_audio(seconds=5.0)
            save_evidence_audio(recent, "audio_activity")

        cv2.putText(frame, f"{label_disp} ({conf_disp:.2f})", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0) if label_disp=="center" else (0,255,255), 2)

        if flash_counter > 0:
            frame = add_red_flash(frame, flash_text)
            flash_counter -= 1

        video_buf.append(frame.copy())
        cv2.imshow("Exam Gaze Proctor — press q to quit", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    audio_mon.stop()

if __name__ == "__main__":
    main()
