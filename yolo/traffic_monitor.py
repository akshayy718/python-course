from ultralytics import YOLO
import cv2
import os

# ---------------- SETTINGS (change these) ----------------
VIDEO_SOURCE = "traffic.mp4"       # video file name in the same folder, or 0 for webcam
MODEL_NAME = "yolov8n.pt"          # fast. "yolov8s.pt" is more accurate but much slower on CPU
IMG_SIZE = 640                     # 640 = fast. 800 or 960 = sees small bikes better but slower
FRAME_SKIP = 1                     # 1 = use every frame, 2 = use every 2nd frame (faster)

# COCO class ids: 1=bicycle, 2=car, 3=motorcycle/scooter, 5=bus, 7=truck
VEHICLE_CLASSES = [1, 2, 3, 5, 7]
TWO_WHEELER_CLASSES = [1, 3]       # these boxes are labeled "Two-Wheeler"

CONF_VEHICLE = 0.30                # minimum confidence for cars, buses, trucks
CONF_TWO_WHEELER = 0.15            # lower for two-wheelers, they are harder to detect

SHOW_PEOPLE = False                # True = TEST MODE: also draw blue boxes on people (not counted)

LOW_LIMIT = 5                      # fewer than this many vehicles = Low traffic
HIGH_LIMIT = 15                    # this many or more = High traffic
LINE_Y = 300                       # height (in pixels) of the counting line
DELAY = 1                          # milliseconds between frames (bigger = slower playback)
# ---------------------------------------------------------

# 1. Load the YOLO model
model = YOLO(MODEL_NAME)

# 2. Find the video file next to this script (or use the webcam)
if VIDEO_SOURCE == 0:
    source = 0
else:
    script_folder = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(script_folder, VIDEO_SOURCE)

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("Could not open video source:", source)
    exit()

# 3. Classes to ask YOLO for (add person only in test mode)
classes_to_detect = VEHICLE_CLASSES + [0] if SHOW_PEOPLE else VEHICLE_CLASSES

# 4. Memory for counting
counted_ids = set()   # IDs of vehicles already counted
previous_y = {}       # vehicle ID -> its height in the previous frame

while True:
    # Skip frames if FRAME_SKIP is bigger than 1 (makes it faster)
    for _ in range(FRAME_SKIP - 1):
        cap.grab()

    ret, frame = cap.read()
    if not ret:
        break  # video ended

    frame = cv2.resize(frame, (960, 540))

    # 5. Detect AND track (tracking gives each vehicle an ID)
    results = model.track(
        frame,
        persist=True,
        classes=classes_to_detect,
        conf=CONF_TWO_WHEELER,
        imgsz=IMG_SIZE,
        verbose=False,
    )
    boxes = results[0].boxes

    annotated = frame.copy()
    detections = []   # (x1, y1, x2, y2, track_id) of vehicles we keep

    # 6. Go through every detection
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        cls = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        track_id = int(boxes.id[i]) if boxes.id is not None else None

        # Test mode: draw people in blue, but do not count them
        if cls == 0:
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
            cv2.putText(annotated, f"Person {conf:.2f}", (int(x1), max(int(y1) - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            continue

        is_two = cls in TWO_WHEELER_CLASSES

        # cars, buses, trucks need higher confidence than two-wheelers
        if not is_two and conf < CONF_VEHICLE:
            continue

        if is_two:
            label = "Two-Wheeler"
            box_color = (0, 255, 255)   # yellow
        else:
            label = model.names[cls].capitalize()
            box_color = (0, 255, 0)     # green

        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
        cv2.putText(annotated, f"{label} {conf:.2f}", (int(x1), max(int(y1) - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        detections.append((x1, y1, x2, y2, track_id))

    # 7. Vehicles visible in this frame right now
    current_count = len(detections)

    # 8. Count a vehicle when it crosses the line (in either direction)
    for x1, y1, x2, y2, track_id in detections:
        if track_id is None:
            continue
        center_y = int((y1 + y2) / 2)

        if track_id in previous_y:
            old_y = previous_y[track_id]
            crossed_down = old_y < LINE_Y <= center_y
            crossed_up = old_y > LINE_Y >= center_y
            if crossed_down or crossed_up:
                counted_ids.add(track_id)

        previous_y[track_id] = center_y

    total_count = len(counted_ids)

    # 9. Traffic density level
    if current_count < LOW_LIMIT:
        density = "LOW"
        color = (0, 200, 0)       # green (BGR)
    elif current_count < HIGH_LIMIT:
        density = "MEDIUM"
        color = (0, 165, 255)     # orange
    else:
        density = "HIGH"
        color = (0, 0, 255)       # red

    # 10. Draw the counting line and the info panel
    cv2.line(annotated, (0, LINE_Y), (960, LINE_Y), (255, 255, 0), 2)
    cv2.rectangle(annotated, (0, 0), (340, 110), (0, 0, 0), -1)
    cv2.putText(annotated, f"In frame: {current_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated, f"Total counted: {total_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated, f"Traffic: {density}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 11. Show the live window
    cv2.imshow("Smart Traffic Monitoring", annotated)

    # Press q to quit
    if cv2.waitKey(DELAY) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()