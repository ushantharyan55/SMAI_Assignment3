 Squat Form Checker
> \*\*AI-powered real-time squat analysis, rep counting, and coaching — in your browser.\*\*
![Streamlit App](https://share.streamlit.io)
![Python 3.9+](https://www.python.org/)
![MediaPipe](https://developers.google.com/mediapipe)
---
 Overview
Squat Form Checker turns any laptop or phone into a virtual fitness coach. Using MediaPipe Pose Landmarker Heavy and real-time joint angle analysis, it evaluates squat depth, bilateral symmetry, hip hinge quality, and top lockout — providing instant, structured feedback on every rep.
Unlike basic fitness apps that only count reps, this tool scores quality of movement and tells you why a rep was counted or rejected.
---
 Features
Feature	Details
Live Webcam Analysis	Real-time skeleton overlay, HUD with joint angles, per-rep flash banners
Video Upload	Frame-by-frame analysis of recorded workouts with session summary
Image Upload	Static posture check with annotated pose and joint metrics
4-Category Scoring	Perfect / Good / Partial / Not a Squat with coaching tips
3-State Rep Tracking	Finite state machine prevents double-counting
Symmetry Detection	Flags lunges and unbalanced movements
---
 Live Demo
▶️ Try it on Streamlit Community Cloud →
> Position your camera to the \*\*side\*\* for most accurate knee-angle readings.
---
 Tech Stack
```
Frontend      │ Streamlit
Pose Detection│ MediaPipe Pose Landmarker Heavy (33 landmarks, \~25 MB)
Vision        │ OpenCV
Math          │ NumPy
Images        │ Pillow (PIL)
```
---
 Installation
```bash
# 1. Clone
git clone https://github.com/yourusername/squat-form-checker.git
cd squat-form-checker

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\\Scripts\\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run SQT.py
```
> The MediaPipe model (\~25 MB) is downloaded automatically on first launch.
---
Repository Structure
```
squat-form-checker/
├── README.md               # Main Streamlit application
├── requirements.txt        # Python dependencies
├── SQT.py
```
---
 How It Works
1. Pose Detection
Every frame is passed through MediaPipe Pose Landmarker (Heavy variant) which returns 33 body landmarks. Only 8 are used: shoulders, hips, knees, and ankles.
2. Angle Calculation
Joint angles are computed via the dot-product formula on 2D vectors:
```
θ = arccos( (BA · BC) / (|BA| × |BC|) )
```
Knee angle: hip → knee → ankle  
Hip angle: shoulder → hip → knee
3. State Machine
```
Standing (>155°) → Partial (110°–140°) → Squatting (<110°) → Standing
```
A rep is counted only on the `Down → Up` transition.
4. Classification
Category	Counted	Criteria
🔥 Perfect Squat	✅	Depth < 100°, symmetric, good hinge, full lockout
✅ Good Squat	✅	Depth < 110°, symmetric, minor form issues
⚠️ Partial Squat	❌	Symmetric but insufficient depth
❌ Not a Squat	❌	Asymmetric leg movement
---
 Screenshots
Live Webcam	Video Analysis
![webcam](assets/screenshot_webcam.png)	![video](assets/screenshot_video.png)
---
 Camera Tips
Side view gives the most accurate knee angles
Front-facing cameras underestimate squat depth by ~10–15°
Ensure your full body is visible in the frame
Good lighting significantly improves landmark detection
---
 Acknowledgements
MediaPipe — Google's pose estimation framework
Streamlit — rapid ML app framework
OpenCV — computer vision library
