import os
import re
import pandas as pd

label_dir = "output/exp1/labels"
output_csv = "output/exp1/results.csv"
data = []

for file_name in os.listdir(label_dir):
    if file_name.endswith(".txt"):
        # Extract frame number using regex
        match = re.search(r"(\d+)\.txt$", file_name)
        if match:
            frame_id = int(match.group(1))
            with open(os.path.join(label_dir, file_name), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 6:
                        class_id, track_id, x, y, w, h = parts
                        data.append([
                            frame_id,
                            float(track_id),  # ⬅ store as float to avoid error
                            float(x),
                            float(y),
                            float(w),
                            float(h)
                        ])

df = pd.DataFrame(data, columns=["frame", "track_id", "x_center", "y_center", "width", "height"])
df.sort_values(by=["track_id", "frame"], inplace=True)
df.to_csv(output_csv, index=False)

print(f"✅ CSV saved to {output_csv}")
