import cv2
from ultralytics import YOLO

class VehicleTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.zone_y_min = 400
        self.zone_y_max = 600
        self.counted_ids = set()
        self.class_counts = {"Car": 0, "Motorcycle": 0, "Truck": 0, "Bus": 0}

    def process_frame(self, frame):
        # Run YOLO tracking
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=640, conf=0.50, verbose=False)
        
        # Safe check: ensure results exist and have boxes
        if not results or not results[0].boxes:
            return frame, len(self.counted_ids), self.class_counts

        boxes = results[0].boxes
        
        # Safe check: ensure tracking IDs exist
        if boxes.id is None:
             return frame, len(self.counted_ids), self.class_counts

        # Extract data cleanly
        xyxys = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        names = results[0].names

        for box, track_id, class_id in zip(xyxys, track_ids, class_ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            class_name = names[class_id].lower() # Ensure lowercase for comparison

            if class_name in ["car", "motorcycle", "truck", "bus"]:
                # Zone Logic
                if self.zone_y_min < cy < self.zone_y_max:
                    if track_id not in self.counted_ids:
                        self.counted_ids.add(track_id)
                        if class_name == "car": self.class_counts["Car"] += 1
                        elif class_name == "motorcycle": self.class_counts["Motorcycle"] += 1
                        elif class_name == "truck": self.class_counts["Truck"] += 1
                        elif class_name == "bus": self.class_counts["Bus"] += 1

                # Drawing (Optimized slightly)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                
                # Format text: Capitalize class name
                label = f"ID:{track_id} {class_name.capitalize()}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw Zone lines
        cv2.line(frame, (0, self.zone_y_min), (frame.shape[1], self.zone_y_min), (255, 0, 0), 2)
        cv2.line(frame, (0, self.zone_y_max), (frame.shape[1], self.zone_y_max), (255, 0, 0), 2)

        return frame, len(self.counted_ids), self.class_counts