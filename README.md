A Computer Vision pipeline for real-time vehicle detection, tracking, and counting using **YOLOv8** and **ByteTrack** algorithm. This project demonstrates the ability to solve real-world traffic analysis problems by implementing a robust "Counting Zone" logic to handle high-speed moving objects.

## 🎯 Features
- **Multi-Class Detection:** Detects and classifies vehicles into 4 categories (`Car`, `Motorcycle`, `Truck`, `Bus`).
- **Object Tracking (ByteTrack):** Assigns persistent IDs to vehicles across frames to prevent duplicate counting.
- **Advanced Counting Logic (Zone-based):** Replaced single-line counting with a Dual-Line "Counting Zone" to accurately catch fast-moving vehicles and drop-frame issues.
- **Model Optimization:** Configured parameters (`conf=0.10`, `iou=0.5`, `imgsz=640`) specifically fine-tuned to recapture small objects (e.g., motorcycles with dark helmets/clothes).
- **Video Exporter:** Automatically renders and exports the annotated `.mp4` file matching the source video's FPS and resolution.

---

## 🏗️ System Architecture & Pipeline

1. **Video Ingestion:** Extracting frames using `cv2.VideoCapture`.
2. **Inference (YOLOv8s):** Processing frames through the YOLOv8 small model to get Bounding Boxes, Classes, and Confidences.
3. **Tracking (ByteTrack):** Associating detections across frames to maintain unique `track_id` for each vehicle.
4. **Logic Calculation:** Extracting the Center Point `(cx, cy)` of the bounding box and checking if it intersects within the predefined Counting Zone (Y-axis: 400px - 600px).
5. **State Management:** Using Python `set()` for $O(1)$ fast lookup to cache counted IDs and update class counters.
6. **Rendering:** Drawing Bounding Boxes, UI Dashboards, and exporting to `.mp4`.

---

## 🛠️ Tech Stack & Dependencies
- **Ultralytics (YOLOv8):** Core inference and tracking engine.
- **OpenCV (`cv2`):** Video I/O, image processing, and UI drawing.
- **Matplotlib:** For initial coordinate plotting and visual debugging.
- **Google Colab (T4 GPU):** Development and inference environment.

---

## 🚀 How to Run (Google Colab / Jupyter)

### 1. Setup Environment
Ensure you have a GPU environment (e.g., Google Colab with T4 GPU) and install the dependencies:
```bash
pip install ultralytics opencv-python-headless matplotlib

2. Prepare Data
Upload your test video into the working directory and rename it to test_video.mp4 (or update the video_path variable in the code).

3. Determine Counting Zone (Optional but Recommended)
Run the coordinate calculation script first to find the optimal Y-axis for your specific camera angle.

import cv2, matplotlib.pyplot as plt
# Load frame, convert BGR to RGB, plot lines, and use plt.imshow() to find coordinates

4. Run the Pipeline
Execute the main script. The system will download yolov8s.pt automatically on the first run.
# The script will process the video frame-by-frame.
# Once finished, 'output_video.mp4' will be saved in your directory.

🧠 Engineering Decisions & Challenges Solved
1. The Motorcycle Detection Problem
Challenge: The baseline yolov8n (Nano) model failed to detect motorcycles, especially those blending into the road (dark clothes/helmets).
Solution: Upgraded to yolov8s (Small) for deeper feature extraction. Forced input size imgsz=640 to retain high-frequency details. Lowered
the confidence threshold to 0.10 to recapture small bounding boxes while using iou=0.5 to handle overlapping vehicles.

2. Fast-Moving Object Misses (The "Jumping" Center Point)
Challenge: Using a single counting line (cy > line_y) caused missed counts if a vehicle moved too fast between frames, skipping the line entirely.
Solution: Implemented a "Counting Zone" (Entry line at Y=400, Exit line at Y=600). The logic was updated to line1_y < cy < line2_y.

3. Real-Time Performance & State
TrackingChallenge: How to prevent the system from counting the same vehicle 30 times (for 30 frames) while it remains in the Counting Zone?
Solution: Utilized a Python set() data structure to cache track_ids. Checking if track_id not in counted_ids provides a lightning-fast O(1) lookup time,
preventing duplicate counts without slowing down the video FPS.

👤 Author
Natee Siriudom 
AI Engineer & Full-Stack Developer
