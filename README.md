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

## Viva notes
**Computer vision**: making computers understand images/video.
**Face vs person detection**: face detection finds only faces; person detection finds whole bodies. A YOLO "person" box is not a face detector.
**YuNet (high level)**: a small CNN scans the image and outputs face boxes with confidence scores and landmarks; boxes below the threshold are dropped.
**Pretrained weights**: numbers a network learned earlier from many labelled images, saved in a file so we only run it (inference).
**BGR / RGB / HSV**: OpenCV loads images as Blue-Green-Red. RGB is the same data in another order. HSV = Hue (colour type), Saturation (purity), Value (brightness); it separates colour from brightness, so thresholding is easier.
**HSV thresholding and mask**: `cv2.inRange` keeps pixels in a range (white=match, black=no) producing a binary mask. Red wraps around hue 0, so it needs two ranges.
**Morphology**: opening (erode then dilate) removes small noise; closing (dilate then erode) fills small holes.
**Contours and boxes**: contours are outlines of white regions; `boundingRect` gives a box. Small areas are ignored as noise.
**Real-time processing**: read a frame, process it, draw, show, repeat in a loop.
**Image processing vs ML**: image processing uses fixed hand-written rules (colour detection here). ML learns rules from data (YuNet).
**Pretrained vs scratch**: pretrained is fast, needs no big dataset or GPU; training from scratch needs lots of data and compute.

**Likely questions**
1. Why HSV instead of BGR? Colour is separated from brightness.
2. Why two ranges for red? Hue wraps at 0/179.
3. Why flip the frame? Mirror view feels natural.
4. What does `waitKey(1)` do? Waits 1 ms for a key and lets the window refresh.
5. Why release the camera? Frees the device for other apps.
6. Why YuNet over Haar? More robust, still CPU-friendly.
7. Is colour detection machine learning? No, rule-based.
8. Why remove small contours? To avoid noise being treated as objects.
9. What limits colour accuracy? Lighting, camera, thresholds, similar-coloured backgrounds.
10. What is the face confidence threshold? Minimum score for accepting a detection.
11. What is a binary mask? Image with only 0 and 255 pixels showing matches.
12. What is a limitation of Haar? Mostly frontal faces, more false positives.
