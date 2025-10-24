🎓 Exam Gaze Proctor

An AI-powered exam proctoring system that detects cheating behavior in real time using:

👀 Eye-gaze tracking (detects if the student is looking away from the screen)

🚨 Multi-face detection (flags when another person appears in frame)

⚡ Red flash warnings in the live feed for instant feedback

📝 Event logging + short evidence video clips of suspicious activity

Built with Python, TensorFlow/Keras, OpenCV, and MediaPipe.

✨ Features

Gaze Classification (CNN)

Trained on a hybrid dataset (Columbia Gaze + custom collected data)

Classes: center, closed, left, right, up, down

Inference collapsed to {center, away, closed} for robust proctoring

Multi-face Detection

Uses MediaPipe FaceMesh to detect >1 person in frame

Logs “multiple faces” events + saves short evidence clips

Red Flash Warnings

Flashes the screen red with a bold warning message when:

Student looks away for too long

Multiple faces are detected

Instant visual feedback for proctor or reviewer

Event Evidence

Logs all events in events/events.log

Saves short .avi video clips as proof of suspicious behavior

🛠️ Tech Stack

Python 3.11

TensorFlow/Keras (CNN training + inference)

OpenCV (video processing, overlays, evidence saving)

MediaPipe (facial landmarks + multi-face detection)

NumPy / Pandas (data handling)

📂 Project Structure
Exam_Proctor/
│
├── raw_images/          # raw dataset (custom + open source)
├── processed/           # preprocessed eye crops
├── scripts/
│   └── capture_dataset.py   # collect custom gaze samples
│
├── preprocess.py        # preprocess dataset -> eye crops + CSV
├── model_train.py       # CNN training script
├── infer_realtime.py    # real-time detection (with red flash + multi-face)
│
├── gaze_cnn.h5          # trained model
├── labels.txt           # class labels
├── requirements.txt
└── README.md

📊 Dataset

Columbia Gaze Dataset (open source)

Custom-collected data (~8,500+ images)

6 classes: center, left, right, up, down, closed

Balanced to avoid bias and improve accuracy

Captured under different lighting/postures for generalization

Current distribution after balancing:
center: ~1400  
closed: ~1280  
down:   ~1470  
up:     ~1520  
left:   ~1880  
right:  ~1920  


🚀 How It Works

Collect Data

Run scripts/capture_dataset.py to capture gaze samples with keyboard controls.

Preprocess

Run python preprocess.py → generates cropped eye images + data.csv.

Train Model

Run python model_train.py → trains CNN on the dataset.

Saves model as gaze_cnn.h5.

Run Real-Time Proctor

Run python infer_realtime.py

Webcam feed opens with overlays, logs suspicious events, flashes warnings, and saves video evidence.

🎥 Demo (Features in Action)

✅ Label overlay → live classification (center, away, closed)

👁️ Eye thumbnails in the corner for debugging

🚨 Red flash warnings for “looking away” or “multiple faces”

📝 Log entries + .avi evidence videos saved in /events/

🔮 Next Improvements (Roadmap)

Add closed-eyes timeout (flag if eyes remain shut >5s).

Subject-wise train/val split for fairer training.

Try lightweight pre-trained models (MobileNetV2/EfficientNet) for better accuracy.

Expand dataset with UT Multiview or MPIIGaze for more diversity.

👨‍💻 Author

Built by Vathsal Upadhyay
 as a college-level project to demonstrate ML/AI fundamentals with a real-world application.