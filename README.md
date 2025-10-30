# Exam Gaze Proctor 

**Exam Gaze Proctor** is an AI-powered system designed to monitor students during online exams.  
It uses **eye gaze tracking**, **face detection**, **audio monitoring**, and **YOLO-based object detection** to flag potentially suspicious behaviors such as looking away, keeping eyes closed for too long, multiple people being present, background voices, or **mobile phone usage**.  

Built with **TensorFlow, OpenCV, MediaPipe, YOLOv8, and sounddevice**, the project demonstrates how deep learning and computer vision can be applied to real-world proctoring scenarios.

---

## ✨ Key Features

### 🔹 Eye Gaze Tracking
- Classifies gaze into three categories:  
  - **Center** → Looking directly at the screen.  
  - **Away** → Looking left, right, up, or down.  
  - **Closed** → Eyes shut.  
- Powered by a custom-trained CNN on both an open dataset and personal calibration images.  
- Uses **temporal smoothing** and **confidence thresholds** to reduce false positives.

---

### 🔹 Closed-Eyes Timeout
- Detects if the user’s eyes remain closed for an extended duration.  
- **If eyes are closed for >5 seconds:**  
  - Red warning overlay is displayed.  
  - Event is logged in `events/events.log`.  
  - A short MP4 clip is saved as evidence.  
- Helps differentiate between natural blinking and suspicious inactivity.

---

### 🔹 Multi-Face Detection
- Ensures only one face is present in the camera feed.  
- If **more than one face** is detected:  
  - Immediate red flash warning (*MULTIPLE FACES DETECTED!*).  
  - Event is logged with a timestamp.  
  - Video evidence is recorded automatically.  
- Useful to prevent collaboration or proxy test-taking.

---

### 🔹 Audio Monitoring
- Listens to the environment through the microphone.  
- If **voices or conversations** are detected for more than 1 second:  
  - Red flash warning (*VOICE DETECTED!*).  
  - Event is logged with a timestamp.  
  - A short **.wav audio clip** is saved in the `events/` folder.

---

### 🔹 Mobile Phone Detection (YOLOv8)
- Integrates **YOLOv8 pretrained model** to detect mobile phones in the webcam feed.  
- If a **cell phone** is detected:  
  - Immediate red flash warning (*MOBILE PHONE DETECTED!*).  
  - Event is logged with a timestamp.  
  - A short MP4 clip is saved as evidence.  
- Prevents students from secretly using their phone during the exam.

---

### 🔹 Real-Time Warnings
- During suspicious activity, the student’s screen flashes **red** with a bold warning message:  
  - *LOOKING AWAY!*  
  - *MULTIPLE FACES DETECTED!*  
  - *EYES CLOSED TOO LONG!*  
  - *VOICE DETECTED!*  
  - *MOBILE PHONE DETECTED!*  
- Provides **instant feedback** to discourage further misconduct.  

---

### 🔹 Evidence Logging
- Every suspicious event is logged in the `events/` folder.  
- **Evidence types saved:**  
  1. A line in `events.log` with the timestamp and event type.  
  2. A short **.mp4 video clip** of the violation (5 seconds leading up to the event).  
  3. For audio events, a **.wav audio clip** is saved.  
- Ensures instructors can review exactly what happened later.

---

## 📂 Project Structure

```
Exam_Proctor/
├── scripts/                # Dataset capture & preprocessing scripts
├── raw_images/             # Personal + open dataset images
├── processed/              # Preprocessed eye crops ready for training
├── model_train.py          # CNN training script
├── infer_realtime.py       # Real-time proctoring system
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── events/                 # Logs + MP4 evidence of suspicious events
```

---

## 🔄 System Pipeline

```mermaid
flowchart TD
    A[Webcam Feed] --> B(MediaPipe Face Mesh)
    B --> C(Eye Crop + Preprocessing)
    C --> D(CNN Gaze Classifier)
    D -->|Center| E1[Normal State]
    D -->|Away| E2[Flag Away]
    D -->|Closed| E3[Closed Eye Timeout Check]

    B --> F(Face Detection - Multi-Face)
    F -->|>1 Face| E4[Flag Multiple Faces]

    H[Microphone Audio] --> I(Audio Monitor)
    I -->|Voices Detected| E5[Flag Voice Detected]

    A --> J[YOLOv8 Object Detection]
    J -->|Cell Phone| E6[Flag Mobile Phone]

    E2 --> K[Red Flash + Log Event]
    E3 --> K
    E4 --> K
    E5 --> K
    E6 --> K

    K --> L[Save Evidence: MP4 + WAV + Log]

```
---

## 📊 Current Capabilities

✅ Eye gaze classification (center / away / closed)

✅ Temporal smoothing for stable predictions

✅ Closed-eyes timeout detection (>5s)

✅ Multi-face detection with warnings

✅ Audio monitoring & voice detection

✅ Mobile phone detection with YOLOv8

✅ Red flash overlay warnings for violations

✅ Evidence logging (event log + MP4 clips + WAV audio)

---

## 🔮 Planned Improvements

⬜ Cheat Behavior Scenarios → add detection for frequent head tilting, use of secondary screen, etc.

⬜ Instructor Dashboard → centralized log and video review system

⬜ Optional Cloud Sync → store violations securely for remote review

⬜ Dataset Expansion → integrate more open datasets to improve CNN generalization

---

👨‍💻 Author

Developed by Vathsal Upadhyay (Vatsy05)
💡 Built as a college-level AI/ML project to showcase practical application of CNNs, MediaPipe, and real-time proctoring.