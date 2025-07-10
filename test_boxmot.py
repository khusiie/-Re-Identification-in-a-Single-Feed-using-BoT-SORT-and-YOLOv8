import cv2
import torch
from boxmot.tracker import BOTSORT  # ✅ correct import
from boxmot.utils import draw_bboxes

# Load YOLOv5 model (your custom model)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')

# Initialize video
video_path = '15sec_input_720p.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize BOTSORT
tracker = BOTSORT()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2, conf, class)

    # Only keep class 0 (players)
    detections = [d[:5] for d in detections if int(d[5]) == 0]

    bboxes = [[*d[:4], d[4]] for d in detections] if detections else []

    # Run BOTSORT
    tracks = tracker.update(bboxes, frame)

    # Draw results
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {int(track_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow('BOTSORT Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
