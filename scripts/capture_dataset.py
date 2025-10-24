# scripts/capture_dataset.py
import cv2, os, time
from pathlib import Path

# 6 gaze classes
CLASSES = ["center","left","right","up","down","closed"]
BASE = os.path.join("..", "raw_images")  # save into raw_images/ outside scripts folder
os.makedirs(BASE, exist_ok=True)
for c in CLASSES:
    Path(os.path.join(BASE, c)).mkdir(parents=True, exist_ok=True)

def banner(frame, text, color=(255,255,255)):
    cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    current = "center"
    saving = False
    delay = 0.15  # seconds between saves while auto-saving
    last = 0

    help_text = "1-6 select class | s toggle save | space capture once | q quit"
    key_map = {ord('1'):0,ord('2'):1,ord('3'):2,ord('4'):3,ord('5'):4,ord('6'):5}

    while True:
        ok, frame = cap.read()
        if not ok: break

        banner(frame, f"class: {current} | saving: {saving}")
        cv2.putText(frame, help_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.imshow("capture_dataset", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        if k in key_map:
            current = CLASSES[key_map[k]]
        if k == ord('s'):
            saving = not saving
            last = 0
        if k == ord(' '):  # capture one frame
            fn = os.path.join(BASE, current, f"{int(time.time()*1000)}.jpg")
            cv2.imwrite(fn, frame)
            print("saved", fn)

        if saving:
            now = time.time()
            if now - last >= delay:
                fn = os.path.join(BASE, current, f"{int(now*1000)}.jpg")
                cv2.imwrite(fn, frame)
                print("saved", fn)
                last = now

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
