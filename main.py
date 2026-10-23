"""Human Face and Colour Detection System (real-time, webcam).

- Face detection : YuNet (pretrained, OpenCV DNN). Falls back to Haar Cascade.
- Colour detection: classic HSV thresholding (no machine learning).
Option 1 (key 1): SINGLE colour detection - pick one colour at a time.
Option 2 (key 2): ALL colours detection   - finds every colour and names the shade.
Single mode: N/P = next/previous colour, R/G/B/Y = Red/Green/Blue/Yellow.
Other keys: M = show mask, Q = quit.
"""
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("Missing packages. Run: python -m pip install -r requirements.txt")
    sys.exit(1)

# ---------------- Settings you can change ----------------
CAMERA_INDEX = 0            # try 1 or 2 if the default camera fails
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_CONTOUR_AREA = 500      # ignore colour blobs smaller than this (pixels)
FACE_SCORE_THRESHOLD = 0.7  # YuNet confidence (0-1); raise to reduce false faces
KERNEL_SIZE = 5             # size of the morphology kernel

# HSV ranges (OpenCV: H 0-179, S 0-255, V 0-255). Each colour may have several ranges.
COLOUR_RANGES = {
    "Red": [((0, 100, 60), (9, 255, 255)), ((170, 100, 60), (179, 255, 255))],
    "Orange": [((10, 100, 101), (17, 255, 255))],
    "Yellow": [((18, 80, 80), (34, 255, 255))],
    "Green": [((35, 40, 40), (84, 255, 255))],
    "Cyan": [((85, 60, 40), (99, 255, 255))],
    "Blue": [((100, 60, 40), (130, 255, 255))],
    "Purple": [((131, 50, 40), (150, 255, 255))],
    "Pink": [((151, 40, 100), (169, 255, 255))],
    "Brown": [((10, 80, 20), (20, 255, 100))],
    "White": [((0, 0, 200), (179, 30, 255))],
    "Gray": [((0, 0, 41), (179, 40, 199))],
    "Black": [((0, 0, 0), (179, 255, 40))],
}
# Webcams often wash colours out. If a colour is missed, lower its saturation (2nd number)
# or value (3rd number) minimum. Click an object to see its HSV value.
# Box colour drawn on screen (BGR) for each colour
DRAW_COLOURS = {
    "Red": (0, 0, 255), "Orange": (0, 140, 255), "Yellow": (0, 255, 255),
    "Green": (0, 200, 0), "Cyan": (255, 255, 0), "Blue": (255, 100, 0),
    "Purple": (200, 0, 160), "Pink": (180, 105, 255), "Brown": (30, 70, 140),
    "White": (255, 255, 255), "Gray": (160, 160, 160), "Black": (80, 80, 80),
}
COLOUR_NAMES = list(COLOUR_RANGES)
# In single-colour mode these keys jump straight to a colour
QUICK_KEYS = {ord("r"): "Red", ord("g"): "Green", ord("b"): "Blue", ord("y"): "Yellow"}

MODELS_DIR = Path(__file__).resolve().parent / "models"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
HAAR_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"
# ----------------------------------------------------------


def load_face_detector():
    """Return (detector, kind). Tries YuNet first, then Haar Cascade."""
    if YUNET_PATH.exists() and hasattr(cv2, "FaceDetectorYN_create"):
        try:
            detector = cv2.FaceDetectorYN_create(
                str(YUNET_PATH), "", (FRAME_WIDTH, FRAME_HEIGHT), FACE_SCORE_THRESHOLD
            )
            return detector, "YuNet"
        except cv2.error as error:
            print("Could not load YuNet:", error)
    else:
        print("YuNet model not found (run: python download_model.py). Using Haar fallback.")

    if HAAR_PATH.exists():
        cascade = cv2.CascadeClassifier(str(HAAR_PATH))
        if not cascade.empty():
            return cascade, "Haar"
    print("ERROR: no face model could be loaded. Check the models/ folder.")
    sys.exit(1)


def detect_faces(frame, detector, kind):
    """Return a list of (x, y, w, h) face boxes."""
    if kind == "YuNet":
        height, width = frame.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(frame)
        if faces is None:
            return []
        return [tuple(int(v) for v in face[:4]) for face in faces]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return [tuple(int(v) for v in box) for box in boxes]


def draw_faces(frame, faces):
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Human Face", (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def make_colour_mask(frame, colour_name):
    """Convert to HSV, threshold the chosen colour, and clean up noise."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in COLOUR_RANGES[colour_name]:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove small specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small holes
    return mask


def find_colour_boxes(mask):
    """Return bounding boxes of contours bigger than MIN_CONTOUR_AREA."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]


def draw_colour_boxes(frame, boxes, colour_name):
    colour = DRAW_COLOURS[colour_name]
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(frame, colour_name, (x, y + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


def draw_info_panel(frame, face_count, mode_text, kind):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Faces: {face_count} ({kind})   {mode_text}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "1=Single colour  2=All colours  N/P=Next/Prev  R G B Y  M=Mask  Q=Quit",
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, hsv_probe["text"], (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


# ---------------- Auto colour mode (any colour, named by shade) ----------------
AUTO_MIN_SATURATION = 60    # pixels paler than this are treated as grey/white
AUTO_MIN_VALUE = 50         # pixels darker than this are treated as black
AUTO_SCALE = 0.5            # process a smaller frame for speed (boxes are scaled back)
HUE_BIN_WIDTH = 12          # groups similar hues so each object becomes its own region


def name_colour(h, s, v):
    """Turn an OpenCV HSV value (H 0-179, S/V 0-255) into a colour name."""
    if v < 40:
        return "Black"
    if s < 40:
        return "White" if v > 200 else "Gray"
    deg = h * 2  # hue in degrees 0-360
    if deg < 15 or deg >= 345:
        if v < 120:
            return "Maroon"
        return "Pink" if (s < 120 and v > 180) else "Red"
    if deg < 40:
        if s < 110 and v > 180:
            return "Beige"
        return "Brown" if v < 150 else "Orange"
    if deg < 70:
        return "Olive" if v < 150 else "Yellow"
    if deg < 95:
        return "Lime"
    if deg < 165:
        return "Dark Green" if v < 110 else "Green"
    if deg < 195:
        return "Teal" if v < 130 else "Cyan"
    if deg < 255:
        if v < 120:
            return "Navy"
        return "Light Blue" if (s < 110 and v > 200) else "Blue"
    if deg < 290:
        return "Lavender" if (s < 110 and v > 200) else "Purple"
    if deg < 330:
        return "Magenta"
    return "Pink"


def auto_detect_colours(frame):
    """Find coloured regions of ANY colour. Returns (regions, mask).
    regions = list of (x, y, w, h, colour_name, bgr_colour)."""
    full_size = (frame.shape[1], frame.shape[0])
    small = cv2.resize(frame, None, fx=AUTO_SCALE, fy=AUTO_SCALE, interpolation=cv2.INTER_AREA)
    min_area = MIN_CONTOUR_AREA * AUTO_SCALE * AUTO_SCALE
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    kernel = np.ones((3, 3), np.uint8)

    # Group pixels into masks: one per hue bin (chromatic) + white/gray/black
    shifted_hue = (hue.astype(np.int32) + HUE_BIN_WIDTH // 2) % 180  # keeps red in one bin
    chromatic = (sat >= AUTO_MIN_SATURATION) & (val >= AUTO_MIN_VALUE)
    groups = []
    for i in range(180 // HUE_BIN_WIDTH):
        groups.append(chromatic & (shifted_hue // HUE_BIN_WIDTH == i))
    groups.append((val < 40))                                  # black
    groups.append((sat < 40) & (val >= 200))                   # white
    groups.append((sat < 40) & (val >= 40) & (val < 200))      # gray

    regions = []
    all_masks = np.zeros(hsv.shape[:2], np.uint8)
    for group in groups:
        mask = group.astype(np.uint8) * 255
        if cv2.countNonZero(mask) < min_area:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        all_masks |= mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            region_mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(region_mask, [contour - [x, y]], -1, 255, -1)
            pixels = hsv[y:y + h, x:x + w][region_mask > 0]
            median_h, median_s, median_v = (int(v) for v in np.median(pixels, axis=0))
            name = name_colour(median_h, median_s, median_v)
            bgr = tuple(int(c) for c in cv2.cvtColor(
                np.uint8([[[median_h, median_s, median_v]]]), cv2.COLOR_HSV2BGR)[0, 0])
            scale = 1 / AUTO_SCALE
            regions.append((int(x * scale), int(y * scale), int(w * scale), int(h * scale), name, bgr))
    return regions, cv2.resize(all_masks, full_size, interpolation=cv2.INTER_NEAREST)


def draw_auto_regions(frame, regions):
    for x, y, w, h, name, bgr in regions:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 4)   # dark outline for contrast
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = min(y + h + th + 8, frame.shape[0] - 2)
        cv2.rectangle(frame, (x, label_y - th - 6), (x + tw + 8, label_y + 4), bgr, -1)
        brightness = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
        text_colour = (0, 0, 0) if brightness > 130 else (255, 255, 255)
        cv2.putText(frame, name, (x + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_colour, 2)


hsv_probe = {"text": "Click on an object to read its HSV", "frame": None}


def on_mouse(event, x, y, flags, param):
    """Left-click shows the HSV value of the clicked pixel (helps tune thresholds)."""
    frame = hsv_probe["frame"]
    if event == cv2.EVENT_LBUTTONDOWN and frame is not None:
        h, s, v = cv2.cvtColor(frame[y:y + 1, x:x + 1], cv2.COLOR_BGR2HSV)[0, 0]
        hsv_probe["text"] = f"Clicked HSV = ({h}, {s}, {v})"
        print(hsv_probe["text"])


def main():
    detector, kind = load_face_detector()
    print("Face detector in use:", kind)

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print(f"ERROR: cannot open camera {CAMERA_INDEX}. Close other apps using it, "
              "check Windows camera privacy settings, or change CAMERA_INDEX in main.py.")
        sys.exit(1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    window = "Human Face and Colour Detection"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    mode = "all"             # "single" (option 1) or "all" (option 2)
    single_colour = "Red"    # colour used in single mode
    show_mask = False
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("ERROR: failed to read a frame from the camera.")
                break
            frame = cv2.flip(frame, 1)  # mirror view

            faces = detect_faces(frame, detector, kind)
            draw_faces(frame, faces)

            if mode == "all":
                regions, mask = auto_detect_colours(frame)
                draw_auto_regions(frame, regions)
                mode_text = "Mode 2: ALL colours"
            else:
                mask = make_colour_mask(frame, single_colour)
                draw_colour_boxes(frame, find_colour_boxes(mask), single_colour)
                mode_text = f"Mode 1: SINGLE colour = {single_colour}"

            hsv_probe["frame"] = frame.copy()
            if show_mask:  # debug window: white = pixels matching the colour(s)
                cv2.imshow("Colour mask (M to hide)", mask)
            draw_info_panel(frame, len(faces), mode_text, kind)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            key = ord(chr(key).lower()) if 65 <= key <= 90 else key  # treat capitals as lowercase
            if key == ord("1"):
                mode = "single"
            elif key == ord("2"):
                mode = "all"
            elif key == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("Colour mask (M to hide)")
            elif mode == "single":
                index = COLOUR_NAMES.index(single_colour)
                if key == ord("n"):
                    single_colour = COLOUR_NAMES[(index + 1) % len(COLOUR_NAMES)]
                elif key == ord("p"):
                    single_colour = COLOUR_NAMES[(index - 1) % len(COLOUR_NAMES)]
                elif key in QUICK_KEYS:
                    single_colour = QUICK_KEYS[key]
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()