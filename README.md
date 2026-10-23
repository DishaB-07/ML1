# Human Face and Colour Detection System

Real-time webcam app (Python + OpenCV) that detects **human faces** and a **selected colour** (Red, Green, Blue, Yellow) at the same time.

## Folder structure
```
human_colour_detection/
├── main.py                 # the application
├── download_model.py       # downloads the YuNet face model
├── requirements.txt
├── .gitignore
├── README.md
└── models/
    └── haarcascade_frontalface_default.xml   # bundled fallback face model
```
`models/face_detection_yunet_2023mar.onnx` is added by `download_model.py` (step 4).

## Face model used
- **Preferred: YuNet** (pretrained CNN face detector from the OpenCV Zoo, MIT licence, ~230 KB, runs on CPU via `cv2.FaceDetectorYN`). Better than Haar with tilted faces, different lighting and partial angles.
- **Fallback: Haar Cascade** (OpenCV's pretrained classical detector, bundled). Used automatically if the YuNet file is missing.
- This is a *face* detector, not a person/body detector like generic YOLO.

## Setup on Windows (VS Code)
1. Install Python 3.9+ from python.org (tick **Add Python to PATH**).
2. Extract the ZIP, open the `human_colour_detection` folder in VS Code (File > Open Folder).
3. Open the terminal (Ctrl + `) and run:
```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```
   If activation is blocked: `Set-ExecutionPolicy -Scope Process Bypass`, then activate again.
4. Download the YuNet model (needs internet, once):
```
python download_model.py
```
5. Run:
```
python main.py
```
6. In VS Code: Ctrl+Shift+P > **Python: Select Interpreter** > choose `.venv`.

## Controls
`1` Red, `2` Green, `3` Blue, `4` Yellow, `Q` Quit. The top bar shows the face count, the detector in use, and the selected colour.

## Settings (top of `main.py`)
`CAMERA_INDEX`, `MIN_CONTOUR_AREA`, `FACE_SCORE_THRESHOLD`, and `COLOUR_RANGES` (HSV thresholds).

## Troubleshooting
- **Cannot open camera**: close Zoom/Teams/Camera app; Windows Settings > Privacy > Camera > allow desktop apps; try `CAMERA_INDEX = 1`.
- **Top bar says Haar**: YuNet file missing. Run `python download_model.py`, or download `face_detection_yunet_2023mar.onnx` from https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet into `models/`.
- **`cv2` not found**: venv not activated or interpreter not selected.
- **Colour not detected / wrong colour**: lighting matters. Adjust the HSV ranges or lower `MIN_CONTOUR_AREA`. Skin and wood can trigger Red/Yellow.
- **Too many false faces**: raise `FACE_SCORE_THRESHOLD` (e.g. 0.85).
