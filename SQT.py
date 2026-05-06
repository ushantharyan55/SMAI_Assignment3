import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import tempfile
import os
import time
import urllib.request
from PIL import Image

st.set_page_config(
    page_title="Squat Form Checker",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)

@st.cache_resource(show_spinner=False)
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading MediaPipe pose model (~25 MB, once only)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

def draw_landmarks(frame, landmarks, W, H):
    pts = [(int(lm.x * W), int(lm.y * H)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, (0, 255, 180), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (0, 200, 255),  1, cv2.LINE_AA)


class SquatAnalyzer:

    SQUAT_DOWN_THRESH = 110
    STAND_UP_THRESH   = 155
    PARTIAL_THRESH    = 140

    def __init__(self, model_path: str, mode: str = "image"):
        BaseOptions        = mp_python.BaseOptions
        PoseLandmarker     = mp_vision.PoseLandmarker
        PoseLandmarkerOpts = mp_vision.PoseLandmarkerOptions
        VisionRunningMode  = mp_vision.RunningMode

        mode_map = {
            "image": VisionRunningMode.IMAGE,
            "video": VisionRunningMode.VIDEO,
            "live":  VisionRunningMode.LIVE_STREAM,
        }

        opts_kwargs = dict(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mode_map[mode],
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._live_result = None
        if mode == "live":
            def _cb(result, _img, _ts):
                self._live_result = result
            opts_kwargs["result_callback"] = _cb

        self.landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOpts(**opts_kwargs)
        )
        self.mode = mode
        self.reset_state()

    def reset_state(self):
        self.rep_count           = 0
        self.stage               = "up"
        self.movement_symmetric  = True
        self.rep_qualities       = []
        self.current_rep_frames  = []
        self.feedback_history    = []

    def close(self):
        self.landmarker.close()

    @staticmethod
    def calc_angle(a, b, c):
        a  = np.array(a, dtype=np.float64)
        b  = np.array(b, dtype=np.float64)
        c  = np.array(c, dtype=np.float64)
        ba = a - b
        bc = c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom < 1e-9:
            return 180.0
        cos_val = np.dot(ba, bc) / denom
        return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

    def extract_angles(self, landmarks, W, H):
        def pt(i):
            lm = landmarks[i]
            return [float(lm.x) * W, float(lm.y) * H]

        left_knee  = self.calc_angle(pt(23), pt(25), pt(27))
        right_knee = self.calc_angle(pt(24), pt(26), pt(28))
        left_hip   = self.calc_angle(pt(11), pt(23), pt(25))
        right_hip  = self.calc_angle(pt(12), pt(24), pt(26))

        return {
            "left_knee":  left_knee,
            "right_knee": right_knee,
            "left_hip":   left_hip,
            "right_hip":  right_hip,
            "avg_knee":   (left_knee + right_knee) / 2.0,
            "avg_hip":    (left_hip  + right_hip)  / 2.0,
            "knee_diff":  abs(left_knee - right_knee),
        }

    @staticmethod
    def is_valid_squat_movement(angles):

        lk   = float(angles["left_knee"])
        rk   = float(angles["right_knee"])
        diff = abs(lk - rk)

        both_bending = lk < 150 and rk < 150
        symmetric    = diff < 45
        return both_bending and symmetric

    def classify_squat(self, min_left_knee, min_right_knee,
                       min_hip, max_knee, was_symmetric):

        min_left_knee  = float(min_left_knee)
        min_right_knee = float(min_right_knee)
        min_hip        = float(min_hip)
        max_knee       = float(max_knee)
        avg_min_knee   = (min_left_knee + min_right_knee) / 2.0

        if not was_symmetric:
            return "not_a_squat", [
                "❌ Not a Squat — asymmetric leg movement detected",
                "💡 Both legs must bend equally. This looks like a lunge, "
                "single-leg squat, or unbalanced position.",
                f"   Left knee: {min_left_knee:.0f}°  |  Right knee: {min_right_knee:.0f}°",
            ]

        if avg_min_knee > self.SQUAT_DOWN_THRESH:
            return "partial_squat", [
                "⚠️ Partial Squat — did not reach full depth",
                f"💡 Your knees only reached {avg_min_knee:.0f}°. "
                "Squat deeper until both knees go below 110° "
                "(note: camera angle can affect readings by ~10–15°).",
            ]

        issues = []

        if avg_min_knee >= 100:
            issues.append(
                f"💡 Depth: knees at {avg_min_knee:.0f}° — go a little deeper for a perfect squat"
            )

        if min_hip > 125:
            issues.append("💡 Hip hinge limited — sit back more and hinge at your hips")
        elif min_hip > 110:
            issues.append("💡 Hip hinge: try sitting back a little more")

        if max_knee < 155:
            issues.append("💡 Not fully standing up — lock out your knees at the top")

        if not issues:
            return "perfect_squat", [
                "🔥 Perfect Squat — both knees below 90°, excellent form and lockout!",
                f"   Left knee: {min_left_knee:.0f}°  |  Right knee: {min_right_knee:.0f}°",
            ]
        else:
            return "good_squat", [
                "✅ Good Squat — reached full depth with minor form notes:",
            ] + issues + [
                f"   Left knee: {min_left_knee:.0f}°  |  Right knee: {min_right_knee:.0f}°",
            ]

    def update_rep_state(self, angles):
        
        avg_knee = float(angles["avg_knee"])

        event_type = None
        feedback   = []

        if self.stage == "up":
            if avg_knee < self.PARTIAL_THRESH:
                self.stage = "partial" if avg_knee >= self.SQUAT_DOWN_THRESH else "down"
                self.current_rep_frames = [angles.copy()]
                self.movement_symmetric = self.is_valid_squat_movement(angles)

        elif self.stage == "partial":
            self.current_rep_frames.append(angles.copy())

            if avg_knee < self.PARTIAL_THRESH:
                if not self.is_valid_squat_movement(angles):
                    self.movement_symmetric = False

            if avg_knee < self.SQUAT_DOWN_THRESH:
                self.stage = "down"

            elif avg_knee > self.STAND_UP_THRESH:
                frames   = self.current_rep_frames
                min_lk   = float(min(f["left_knee"]  for f in frames))
                min_rk   = float(min(f["right_knee"] for f in frames))
                min_h    = float(min(f["avg_hip"]    for f in frames))
                max_knee = float(max(f["avg_knee"]   for f in frames))

                event_type, feedback = self.classify_squat(
                    min_lk, min_rk, min_h, max_knee,
                    self.movement_symmetric
                )
                self.feedback_history.append((event_type, feedback))
                self._reset_rep_buffer()

        elif self.stage == "down":
            self.current_rep_frames.append(angles.copy())
            if avg_knee < self.PARTIAL_THRESH:
                if not self.is_valid_squat_movement(angles):
                    self.movement_symmetric = False

            if avg_knee > self.STAND_UP_THRESH:
                frames = self.current_rep_frames
                min_lk = float(min(f["left_knee"]  for f in frames))
                min_rk = float(min(f["right_knee"] for f in frames))
                min_h  = float(min(f["avg_hip"]    for f in frames))
                max_knee = float(max(f["avg_knee"] for f in frames))

                event_type, feedback = self.classify_squat(
                    min_lk, min_rk, min_h, max_knee,
                    self.movement_symmetric
                )

                if event_type in ("perfect_squat", "good_squat"):
                    self.rep_count += 1
                    self.rep_qualities.append(
                        100 if event_type == "perfect_squat" else 75
                    )

                self.feedback_history.append((event_type, feedback))
                self._reset_rep_buffer()

        return event_type, feedback

    def _reset_rep_buffer(self):
        self.current_rep_frames  = []
        self.stage               = "up"
        self.movement_symmetric  = True

    def draw_hud(self, frame, angles, event_type=None, quality=None):
        H, W = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (295, 195), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (10, 10), (295, 195), (0, 200, 255), 1)

        def txt(msg, x, y, size=0.55, color=(230, 230, 230), bold=False):
            cv2.putText(frame, msg, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, size, color,
                        2 if bold else 1, cv2.LINE_AA)

        stage_labels = {"up": "STANDING", "down": "SQUATTING", "partial": "PARTIAL BEND"}
        stage_colors = {"up": (100, 255, 100), "down": (0, 200, 255), "partial": (255, 180, 0)}
        stage_lbl = stage_labels.get(self.stage, self.stage.upper())
        stage_col = stage_colors.get(self.stage, (200, 200, 200))

        txt("SQUAT FORM CHECKER", 20, 38,  0.75, (0, 200, 255), bold=True)
        txt(f"REPS : {self.rep_count}", 20, 68,  0.65, (255, 255, 255), bold=True)
        txt(f"STAGE: {stage_lbl}", 20, 95,  0.48, stage_col)
        txt(f"L KNEE: {angles['left_knee']:.0f} deg", 20, 120, 0.46)
        txt(f"R KNEE: {angles['right_knee']:.0f} deg", 20, 141, 0.46)
        txt(f"L HIP : {angles['left_hip']:.0f} deg", 20, 162, 0.46)

        if event_type == "perfect_squat":
            cv2.rectangle(frame, (W//2-170, H//2-44), (W//2+170, H//2+44), (50, 205, 50), -1)
            cv2.putText(frame, f"REP {self.rep_count}  --  PERFECT SQUAT",
                        (W//2-155, H//2+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
        elif event_type == "good_squat":
            cv2.rectangle(frame, (W//2-170, H//2-44), (W//2+170, H//2+44), (144, 238, 50), -1)
            cv2.putText(frame, f"REP {self.rep_count}  --  GOOD SQUAT",
                        (W//2-155, H//2+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
        elif event_type == "partial_squat":
            cv2.rectangle(frame, (W//2-170, H//2-44), (W//2+170, H//2+44), (0, 140, 255), -1)
            cv2.putText(frame, "PARTIAL SQUAT  --  NOT COUNTED",
                        (W//2-155, H//2+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        elif event_type == "not_a_squat":
            cv2.rectangle(frame, (W//2-170, H//2-44), (W//2+170, H//2+44), (0, 50, 200), -1)
            cv2.putText(frame, "NOT A SQUAT  --  NOT COUNTED",
                        (W//2-155, H//2+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def process_frame(self, frame_bgr, timestamp_ms: int = 0):
        H, W = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb  = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            if self.mode == "image":
                result = self.landmarker.detect(mp_image)
            elif self.mode == "video":
                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            else:
                self.landmarker.detect_async(mp_image, timestamp_ms)
                time.sleep(0.01)
                result = self._live_result
        except Exception:
            return frame_bgr, None, None, []

        if result is None or not result.pose_landmarks:
            return frame_bgr, None, None, [] 

        landmarks  = result.pose_landmarks[0]
        angles     = self.extract_angles(landmarks, W, H)
        event_type, feedback = self.update_rep_state(angles)

        quality = {
            "perfect_squat": "Perfect",
            "good_squat":    "Good",
            "partial_squat": "Partial",
            "not_a_squat":   "Not a Squat",
        }.get(event_type, None)

        annotated = frame_bgr.copy()
        draw_landmarks(annotated, landmarks, W, H)
        annotated = self.draw_hud(annotated, angles, event_type, quality)
        return annotated, angles, event_type, feedback



def show_feedback(feedback_list):
    for msg in feedback_list:
        if msg.startswith("✅") or msg.startswith("🔥"):
            st.success(msg)
        elif msg.startswith("💡"):
            st.info(msg)
        elif msg.startswith("⚠️"):
            st.warning(msg)
        elif msg.startswith("❌"):
            st.error(msg)
        else:
            st.write(msg)


if "run_webcam" not in st.session_state:
    st.session_state.run_webcam = False
if "webcam_analyzer" not in st.session_state:
    st.session_state.webcam_analyzer = None
    
model_path = download_model()

with st.sidebar:
    st.title("🏋️ Squat Form Checker")
    #st.caption("AI-powered squat form analyzer")
    st.divider()

    st.subheader("📐 Angle Thresholds")
    st.markdown("""
    | Stage | Knee Angle |
    |---|---|
    | Perfect squat | < 100° |
    | Good squat | 100° – 110° |
    | Partial bend | 110° – 140° |
    | Standing | > 155° |
    """)

    st.subheader("🔢 Rep Counting Rules")
    st.markdown("""
    - 🔥 **Perfect Squat** — counted, full depth + perfect form
    - ✅ **Good Squat** — counted, full depth + minor issues
    - ⚠️ **Partial Squat** — NOT counted, symmetric but shallow
    - ❌ **Not a Squat** — NOT counted, asymmetric (lunge / one-leg)
    """)

st.title("🏋️ Squat Form Checker")
st.caption("Real-time squat analysis · Rep counting · Form coaching")
st.divider()

tab_webcam, tab_video, tab_image = st.tabs([
    "📷  Live Webcam",
    "🎬  Upload Video",
    "🖼️  Upload Image",
])

with tab_webcam:
    st.subheader("📷 Live Squat Analysis")

    col_start, col_stop, _ = st.columns([1, 1, 4])
    with col_start:
        start_btn = st.button("▶ Start Session", type="primary")
    with col_stop:
        stop_btn = st.button("⏹ Stop Session")

    if start_btn:
        if st.session_state.webcam_analyzer is not None:
            try:
                st.session_state.webcam_analyzer.close()
            except Exception:
                pass
        st.session_state.webcam_analyzer = SquatAnalyzer(model_path, mode="live")
        st.session_state.run_webcam = True

    if stop_btn:
        st.session_state.run_webcam = False

    m1, m2, m3, m4 = st.columns(4)
    rep_ph   = m1.empty()
    stage_ph = m2.empty()
    qual_ph  = m3.empty()
    score_ph = m4.empty()
    frame_ph = st.empty()
    fb_ph    = st.empty()

    def render_metrics(az):
        perfect  = sum(1 for ev, _ in az.feedback_history if ev == "perfect_squat")
        good     = sum(1 for ev, _ in az.feedback_history if ev == "good_squat")
        partial  = sum(1 for ev, _ in az.feedback_history if ev == "partial_squat")
        not_sq   = sum(1 for ev, _ in az.feedback_history if ev == "not_a_squat")
        stage_label = {"up": "Standing", "down": "Squatting", "partial": "Partial Bend"}.get(
                        az.stage, az.stage.title())
        rep_ph.metric("🔢 Valid Reps",      az.rep_count)
        stage_ph.metric("📍 Stage",         stage_label)
        qual_ph.metric("🔥 Perfect / ✅ Good", f"{perfect} / {good}")
        score_ph.metric("⚠️ Partial / ❌ Not", f"{partial} / {not_sq}")

    if st.session_state.run_webcam and st.session_state.webcam_analyzer is not None:
        cap = cv2.VideoCapture(0)
        az  = st.session_state.webcam_analyzer

        if not cap.isOpened():
            st.error("Could not open webcam. Check camera permissions.")
            st.session_state.run_webcam = False
        else:
            st.info("🟢 Webcam active — for best accuracy, position camera at your **side** so both knees are clearly visible.")
            ts = 0
            while st.session_state.run_webcam:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, angles, event_type, feedback = az.process_frame(frame, timestamp_ms=ts)
                ts += 33

                frame_ph.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )
                render_metrics(az)

                if event_type is not None and feedback:
                    with fb_ph.container():
                        if event_type == "perfect_squat":
                            st.subheader(f"🔥 Rep {az.rep_count} — Perfect Squat!")
                        elif event_type == "good_squat":
                            st.subheader(f"✅ Rep {az.rep_count} — Good Squat!")
                        elif event_type == "partial_squat":
                            st.subheader("⚠️ Partial Squat — Not Counted")
                        elif event_type == "not_a_squat":
                            st.subheader("❌ Not a Squat — Not Counted")
                        show_feedback(feedback)

                time.sleep(0.01)

            cap.release()
            st.success("Session ended.")
            render_metrics(az)

            if az.feedback_history:
                st.divider()
                st.subheader("📋 Session History")
                ICONS  = {"perfect_squat": "🔥", "good_squat": "✅",
                          "partial_squat": "⚠️", "not_a_squat": "❌"}
                LABELS = {"perfect_squat": "Perfect Squat — Counted",
                          "good_squat":    "Good Squat — Counted",
                          "partial_squat": "Partial Squat — Not Counted",
                          "not_a_squat":   "Not a Squat — Not Counted"}
                for i, (ev, fb) in enumerate(az.feedback_history):
                    icon  = ICONS.get(ev, "ℹ️")
                    label = LABELS.get(ev, ev)
                    with st.expander(f"Attempt {i+1} — {icon} {label}"):
                        show_feedback(fb)
    else:
        render_metrics(type("D", (), {
            "rep_count": 0, "stage": "up", "feedback_history": []
        })())
        frame_ph.info("Click **▶ Start Session** to begin live analysis.")

with tab_video:
    st.subheader("🎬 Video Squat Analysis")

    uploaded_video = st.file_uploader(
        "Upload a squat video", type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_video:
        skip_n = st.slider(
            "Process every Nth frame", 1, 10, 2,
            help="Higher = faster but slightly less accurate"
        )

        if st.button("🔍 Analyze Video", type="primary"):
            az = SquatAnalyzer(model_path, mode="video")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            cap   = cv2.VideoCapture(tmp_path)
            fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            st.subheader("⏳ Processing...")
            prog       = st.progress(0, text="Starting...")
            frame_ph_v = st.empty()
            vm1, vm2, vm3, vm4 = st.columns(4)

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % int(skip_n) == 0:
                    ts_ms = int(frame_idx / fps * 1000)
                    annotated, _, event_type, _ = az.process_frame(frame, timestamp_ms=ts_ms)
                    frame_ph_v.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )
                    pct = min(frame_idx / max(total, 1), 1.0)
                    prog.progress(
                        pct,
                        text=f"Frame {frame_idx}/{total}  |  Valid reps: {az.rep_count}"
                    )
                    n_p = sum(1 for ev, _ in az.feedback_history if ev == "perfect_squat")
                    n_g = sum(1 for ev, _ in az.feedback_history if ev == "good_squat")
                    n_x = sum(1 for ev, _ in az.feedback_history if ev == "partial_squat")
                    n_n = sum(1 for ev, _ in az.feedback_history if ev == "not_a_squat")
                    stage_lbl = {"up": "Standing", "down": "Squatting", "partial": "Partial"}.get(
                                    az.stage, az.stage.title())
                    vm1.metric("🔢 Valid Reps",       az.rep_count)
                    vm2.metric("📍 Stage",             stage_lbl)
                    vm3.metric("🔥 Perfect / ✅ Good", f"{n_p} / {n_g}")
                    vm4.metric("⚠️ Partial / ❌ Not",  f"{n_x} / {n_n}")

                frame_idx += 1

            cap.release()
            az.close()
            os.unlink(tmp_path)
            prog.progress(1.0, text="Analysis complete!")

            st.divider()
            st.subheader("📊 Session Summary")
            total_attempts = len(az.feedback_history)
            n_perfect  = sum(1 for ev, _ in az.feedback_history if ev == "perfect_squat")
            n_good     = sum(1 for ev, _ in az.feedback_history if ev == "good_squat")
            n_partial  = sum(1 for ev, _ in az.feedback_history if ev == "partial_squat")
            n_not      = sum(1 for ev, _ in az.feedback_history if ev == "not_a_squat")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("🔢 Valid Reps",      az.rep_count)
            s2.metric("📊 Total Attempts",  total_attempts)
            s3.metric("🔥 Perfect Squats",  n_perfect)
            s4.metric("✅ Good Squats",      n_good)

            c1, c2 = st.columns(2)
            c1.metric("⚠️ Partial Squats",  n_partial)
            c2.metric("❌ Not a Squat",      n_not)

            if az.feedback_history:
                st.divider()
                st.subheader("📋 Attempt Breakdown")
                ICONS  = {"perfect_squat": "🔥", "good_squat": "✅",
                          "partial_squat": "⚠️", "not_a_squat": "❌"}
                LABELS = {"perfect_squat": "Perfect Squat — Counted",
                          "good_squat":    "Good Squat — Counted",
                          "partial_squat": "Partial Squat — Not Counted",
                          "not_a_squat":   "Not a Squat — Not Counted"}
                for i, (ev, fb) in enumerate(az.feedback_history):
                    icon  = ICONS.get(ev, "ℹ️")
                    label = LABELS.get(ev, ev)
                    with st.expander(f"Attempt {i+1} — {icon} {label}"):
                        show_feedback(fb)


with tab_image:
    st.subheader("🖼️ Image Pose Assessment")

    uploaded_img = st.file_uploader(
        "Upload a squat photo", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_img:
        pil_img   = Image.open(uploaded_img).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(pil_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)

        az = SquatAnalyzer(model_path, mode="image")
        annotated, angles, _, _ = az.process_frame(frame_bgr)

        col_img, col_info = st.columns([3, 2])

        with col_img:
            st.subheader("Pose Detection")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col_info:
            if angles is not None:
                left_k  = float(angles["left_knee"])
                right_k = float(angles["right_knee"])
                left_h  = float(angles["left_hip"])
                right_h = float(angles["right_hip"])
                avg_k   = float(angles["avg_knee"])
                avg_h   = float(angles["avg_hip"])

                st.subheader("📐 Joint Angles")
                a1, a2 = st.columns(2)
                a1.metric("Left Knee",  f"{left_k:.1f}°")
                a2.metric("Right Knee", f"{right_k:.1f}°")
                a3, a4 = st.columns(2)
                a3.metric("Left Hip",   f"{left_h:.1f}°")
                a4.metric("Right Hip",  f"{right_h:.1f}°")

                st.write("**Knee bend** (lower = deeper squat):")
                st.progress(
                    min(avg_k / 180.0, 1.0),
                    text=f"{avg_k:.1f}° — "
                         f"{'Full squat ✅' if avg_k < 100 else 'Partial 💡' if avg_k < 130 else 'Standing ℹ️'}"
                )
                st.write("**Hip hinge** (lower = more sit-back):")
                st.progress(
                    min(avg_h / 180.0, 1.0),
                    text=f"{avg_h:.1f}°"
                )

                st.divider()
                st.subheader("📋 Form Assessment")

                knee_diff = abs(left_k - right_k)
                is_sym    = left_k < 150 and right_k < 150 and knee_diff < 45

                if avg_k > 155:
                    st.info("ℹ️ Standing position — no squat detected in this image")
                elif not is_sym:
                    st.error("❌ Not a Squat — legs are not bending equally")
                    st.info(
                        f"💡 Left knee: {left_k:.1f}°  |  Right knee: {right_k:.1f}°  "
                        f"|  Difference: {knee_diff:.1f}° (must be < 45°)"
                    )
                    st.info("💡 Both legs must bend symmetrically — this looks like a lunge or single-leg position")
                elif avg_k > 110:
                    st.warning("⚠️ Partial Squat — did not reach full depth")
                    st.info(f"💡 Your knees reached {avg_k:.1f}°. Squat deeper until both go below 110° (camera angle can affect readings by ~10–15°).")
                else:
                    squat_type, feedback = az.classify_squat(
                        left_k, right_k, avg_h, avg_k, True
                    )
                    if squat_type == "perfect_squat":
                        st.success("🔥 Perfect Squat!")
                    else:
                        st.success("✅ Good Squat!")
                    st.divider()
                    st.subheader("💬 Coaching Tips")
                    show_feedback(feedback)

            else:
                st.warning("No person detected. Try a clearer, well-lit photo with your full body visible.")

        az.close()
