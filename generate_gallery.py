import os
import cv2

# Paths
LABELS_DIR = "output/exp1/labels"
VIDEO_PATH = "15sec_input_720p.mp4"
GALLERY_DIR = "output/exp1/gallery"

# Create gallery directory
os.makedirs(GALLERY_DIR, exist_ok=True)

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"❌ Failed to open video {VIDEO_PATH}"

# Load all frames into memory (safe for short clips)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

print(f"📽️ Loaded {len(frames)} frames.")

# Process each label file
for label_file in sorted(os.listdir(LABELS_DIR)):
    if not label_file.endswith('.txt'):
        continue

    try:
        # Extract frame index from file name
        frame_id = int(label_file.split('.')[0].split('_')[-1])
    except ValueError:
        print(f"⚠️ Skipping invalid file: {label_file}")
        continue

    if frame_id >= len(frames):
        print(f"⚠️ Frame {frame_id} not found in video.")
        continue

    frame = frames[frame_id]
    label_path = os.path.join(LABELS_DIR, label_file)

    with open(label_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            parts = line.strip().split()
            if len(parts) != 6:
                continue  # Skip malformed lines

            class_id, x, y, w, h, track_id = parts
            track_id = int(float(track_id))  # Sometimes written as float

            # Convert to pixel coordinates
            H, W, _ = frame.shape
            x, y, w, h = map(float, [x, y, w, h])
            x1 = int((x - w / 2) * W)
            y1 = int((y - h / 2) * H)
            x2 = int((x + w / 2) * W)
            y2 = int((y + h / 2) * H)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Save cropped thumbnail
            crop_path = os.path.join(GALLERY_DIR, f"id_{track_id}_f{frame_id}_{i}.jpg")
            cv2.imwrite(crop_path, crop)

print(f"✅ Gallery thumbnails saved in: {GALLERY_DIR}")
