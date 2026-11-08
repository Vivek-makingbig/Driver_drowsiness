# Driver Drowsiness Detection System

A real-time driver drowsiness detection application built using **Python, OpenCV, Dlib, and Streamlit**. The system uses facial landmarks to monitor eye closure and yawning through a live webcam feed and detects signs of driver fatigue.

The application provides a visual interface for monitoring the driver's condition and generates alerts when drowsiness is detected.

## Features

- **Real-Time Face Detection:** Detects faces from a live webcam stream using OpenCV.
- **Facial Landmark Detection:** Uses Dlib's 68-point facial landmark predictor.
- **Eye Aspect Ratio (EAR):** Monitors eye closure to identify signs of drowsiness.
- **Yawn Detection:** Measures the distance between lip landmarks to detect yawning.
- **Live Video Processing:** Processes webcam frames through Streamlit WebRTC.
- **Visual Alerts:** Displays detection results and drowsiness status through the application interface.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| OpenCV | Image processing and face detection |
| Dlib | Facial landmark detection |
| NumPy | Numerical operations |
| SciPy | Distance calculations |
| imutils | Image processing utilities |
| Streamlit WebRTC | Real-time webcam streaming |
| PyAV | Video frame processing |

## Project Structure

```text
Driver_drowsiness/
│
├── app.py
├── drowsiness_yawn.py
├── requirements.txt
├── Alert.wav
├── haarcascade_frontalface_default.xml
├── shape_predictor_68_face_landmarks.dat
└── README.md
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Vivek-makingbig/Driver_drowsiness.git
```

### 2. Navigate to the Project Directory

```bash
cd Driver_drowsiness
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser. Allow webcam access when prompted.

## How It Works

The system follows these steps:

1. **Video Capture:** Captures live video from the driver's webcam using Streamlit WebRTC.
2. **Face Detection:** Detects the driver's face using the OpenCV Haar Cascade classifier.
3. **Facial Landmark Extraction:** Uses Dlib's 68-point facial landmark predictor to identify important facial regions.
4. **Eye Monitoring:** Calculates the Eye Aspect Ratio (EAR) using eye landmarks to determine whether the eyes are closed.
5. **Yawn Detection:** Calculates the distance between the upper and lower lip landmarks to identify yawning.
6. **Drowsiness Detection:** Monitors eye closure across consecutive frames and identifies potential drowsiness.
7. **Alert Generation:** Displays visual warnings when signs of drowsiness or yawning are detected.

## Important Files

- `app.py`: Main Streamlit application.
- `drowsiness_yawn.py`: Original standalone Python implementation.
- `haarcascade_frontalface_default.xml`: Face detection model.
- `shape_predictor_68_face_landmarks.dat`: Dlib facial landmark prediction model.
- `Alert.wav`: Audio alert file.
- `requirements.txt`: Python dependencies.

## Limitations

- Detection accuracy depends on lighting conditions and webcam quality.
- Extreme head movements and face occlusion may affect landmark detection.
- Spectacles and camera positioning can influence detection performance.
- The system is intended as a prototype and should not be considered a replacement for dedicated automotive safety systems.

## Future Improvements

- Improve alert delivery and audio notifications.
- Enhance detection under different lighting conditions.
- Integrate head-pose estimation for better driver monitoring.
- Explore machine learning models for more robust fatigue detection.
- Develop a mobile-friendly driver monitoring application.

## Acknowledgements

This project is an extension and modification of the original **Real-Time Drowsiness Detection System** by [Anshuman Srivastava](https://github.com/AnshumanSrivastava108/Real-Time-Drowsiness-Detection-System).

The original implementation provided the foundation for facial landmark-based drowsiness detection. This version extends the project with a Streamlit web interface and real-time webcam processing using WebRTC.

## Author

**Vivek Kumar Shaw**

- GitHub: [Vivek-makingbig](https://github.com/Vivek-makingbig)

---

*This project is developed for educational and research purposes.*