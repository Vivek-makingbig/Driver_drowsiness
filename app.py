import streamlit as st
import streamlit.components.v1 as components
import cv2
import dlib
import av
import numpy as np
import imutils
import os

from scipy.spatial import distance as dist
from imutils import face_utils
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode


# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Driver Drowsiness Detection",
      layout="wide"
)

st.title("Driver Drowsiness Detection System")
st.write("Real-time detection of eye closure and yawning using your webcam.")


# --------------------------------------------------
# Load Files
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
ALARM_PATH = os.path.join(BASE_DIR, "Alert.wav")

if not os.path.exists(CASCADE_PATH):
    st.error("Haar Cascade file not found.")
    st.stop()

if not os.path.exists(PREDICTOR_PATH):
    st.error("Dlib shape predictor file not found.")
    st.stop()

if not os.path.exists(ALARM_PATH):
    st.warning("Alert.wav not found. Detection will work without the alarm.")


# --------------------------------------------------
# Constants (Same as Your Original Code)
# --------------------------------------------------

EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 30

YAWN_THRESH = 30
YAWN_CONSEC_FRAMES = 30


# --------------------------------------------------
# Load Detector and Predictor
# --------------------------------------------------

detector = cv2.CascadeClassifier(CASCADE_PATH)
predictor = dlib.shape_predictor(PREDICTOR_PATH)

if detector.empty():
    st.error("Unable to load Haar Cascade detector.")
    st.stop()


# --------------------------------------------------
# Original Functions
# --------------------------------------------------

def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


def final_ear(shape):

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0

    return ear, leftEye, rightEye


def lip_distance(shape):

    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])

    return distance


# --------------------------------------------------
# Video Processing Class
# --------------------------------------------------

class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.COUNTER = 0
        self.alarm_status = False
        self.alarm_status2 = False

        self.drowsy = False
        self.yawning = False

        self.ear = 0
        self.distance = 0

    def recv(self, frame):

        frame = frame.to_ndarray(format="bgr24")

        frame = imutils.resize(frame, width=450)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        self.drowsy = False
        self.yawning = False

        for (x, y, w, h) in rects:

            rect = dlib.rectangle(
                int(x),
                int(y),
                int(x + w),
                int(y + h)
            )

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            eye = final_ear(shape)

            ear = eye[0]
            leftEye = eye[1]
            rightEye = eye[2]

            distance = lip_distance(shape)

            self.ear = ear
            self.distance = distance

            # Draw eye contours

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(
                frame, [leftEyeHull], -1, (0, 255, 0), 1
            )

            cv2.drawContours(
                frame, [rightEyeHull], -1, (0, 255, 0), 1
            )

            # Draw lip contour

            lip = shape[48:60]

            cv2.drawContours(
                frame, [lip], -1, (0, 255, 0), 1
            )

            # --------------------------------------
            # Eye Aspect Ratio Detection
            # --------------------------------------

            if ear < EYE_AR_THRESH:

                self.COUNTER += 1

                if self.COUNTER >= EYE_AR_CONSEC_FRAMES:

                    self.alarm_status = True
                    self.drowsy = True

                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

            else:

                self.COUNTER = 0
                self.alarm_status = False

            # --------------------------------------
            # Yawning Detection
            # --------------------------------------

            if distance > YAWN_THRESH:

                self.alarm_status2 = True
                self.yawning = True

                cv2.putText(
                    frame,
                    "Yawn Alert",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            else:

                self.alarm_status2 = False

            # --------------------------------------
            # Display EAR and YAWN Values
            # --------------------------------------

            cv2.putText(
                frame,
                "EAR: {:.2f}".format(ear),
                (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "YAWN: {:.2f}".format(distance),
                (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        return av.VideoFrame.from_ndarray(frame, format="bgr24")


# --------------------------------------------------
# Start Webcam
# --------------------------------------------------

st.subheader("Live Webcam")

ctx = webrtc_streamer(
    key="drowsiness-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)


# --------------------------------------------------
# Display Detection Status
# --------------------------------------------------

st.subheader("Detection Status")

status_area = st.empty()


@st.fragment(run_every=1)
def show_status():

    if ctx.state.playing and ctx.video_processor:

        processor = ctx.video_processor

        if processor.drowsy:

            status_area.error("DROWSINESS DETECTED! Please take a break.")

        elif processor.yawning:

            status_area.warning("Yawning detected.")

        else:

            status_area.success("No signs of drowsiness detected.")

        st.write(f"EAR Value: {processor.ear:.2f}")
        st.write(f"Yawn Distance: {processor.distance:.2f}")

        # Play the alarm when an alert first appears.
        # The browser may block automatic audio until the user interacts with the page.
        alert_now = processor.drowsy or processor.yawning

        if alert_now and os.path.exists(ALARM_PATH):

            if not st.session_state.get("alarm_was_active", False):

                st.audio(
                    ALARM_PATH,
                    format="audio/wav",
                    autoplay=True
                )

        st.session_state["alarm_was_active"] = alert_now

    else:

        status_area.info("Start the webcam to begin detection.")
        st.session_state["alarm_was_active"] = False


show_status()