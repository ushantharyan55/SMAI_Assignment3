# Squat Form Checker

> **AI-powered real-time squat analysis, rep counting, and coaching — in your browser.**

[Streamlit App](https://share.streamlit.io)
[Python 3.9+](https://www.python.org/)
[MediaPipe](https://developers.google.com/mediapipe)

---

## Overview

Squat Form Checker turns any laptop or phone into a virtual fitness coach. Using **MediaPipe Pose Landmarker Heavy** and real-time joint angle analysis, it evaluates squat depth, bilateral symmetry, hip hinge quality, and top lockout—providing instant, structured feedback on every rep.

Unlike basic fitness apps that only count reps, this tool scores the **quality of movement** and explains why a rep was counted or rejected.

---

## Features

| Feature              | Details                                                                  |
| -------------------- | ------------------------------------------------------------------------ |
| Live Webcam Analysis | Real-time skeleton overlay, HUD with joint angles, per-rep flash banners |
| Video Upload         | Frame-by-frame analysis of recorded workouts with session summary        |
| Image Upload         | Static posture check with annotated pose and joint metrics               |
| 4-Category Scoring   | Perfect / Good / Partial / Not a Squat with coaching tips                |
| 3-State Rep Tracking | Finite state machine prevents double-counting                            |
| Symmetry Detection   | Flags lunges and unbalanced movements                                    |

---

## Live Demo

▶️ Try it on Streamlit Community Cloud →

> Position your camera to the **side** for the most accurate knee-angle readings.

---

## Tech Stack

```text
Frontend       │ Streamlit
Pose Detection │ MediaPipe Pose Landmarker Heavy (33 landmarks, ~25 MB)
Vision         │ OpenCV
Math           │ NumPy
Images         │ Pillow (PIL)
```

---

## Installation


# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
streamlit run SQT.py
```

> The MediaPipe model (~25 MB) is downloaded automatically on first launch.

---

## Repository Structure

```text
squat-form-checker/
├── README.md          # Project documentation
├── requirements.txt   # Python dependencies
├── SQT.py             # Main Streamlit application
```

---

## How It Works

### 1. Pose Detection

Every frame is passed through **MediaPipe Pose Landmarker (Heavy variant)**, which returns 33 body landmarks.

Only 8 landmarks are used for squat analysis:

* Shoulders
* Hips
* Knees
* Ankles

---

### 2. Angle Calculation

Joint angles are computed using the dot-product formula on 2D vectors:

```text
θ = arccos( (BA · BC) / (|BA| × |BC|) )
```

* **Knee angle:** hip → knee → ankle
* **Hip angle:** shoulder → hip → knee

---

### 3. State Machine

```text
Standing (>155°)
   ↓
Partial (110°–140°)
   ↓
Squatting (<110°)
   ↓
Standing
```

A rep is counted only during the **Down → Up** transition.

---

### 4. Classification

| Category         | Counted | Criteria                                          |
| ---------------- | ------- | ------------------------------------------------- |
| 🔥 Perfect Squat | ✅       | Depth < 100°, symmetric, good hinge, full lockout |
| ✅ Good Squat     | ✅       | Depth < 110°, symmetric, minor form issues        |
| ⚠️ Partial Squat | ❌       | Symmetric but insufficient depth                  |
| ❌ Not a Squat    | ❌       | Asymmetric leg movement                           |

---

## Camera Tips

* Side view gives the most accurate knee-angle measurements
* Front-facing cameras may underestimate squat depth by ~10–15%
* Ensure your full body is visible in the frame
* Good lighting significantly improves landmark detection

---

## Acknowledgements

* **MediaPipe** - Google’s pose estimation framework
* **Streamlit** - Rapid ML app framework
* **OpenCV** - Computer vision library
