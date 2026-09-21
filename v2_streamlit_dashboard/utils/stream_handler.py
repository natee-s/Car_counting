import cv2

class VideoStreamer:
    def __init__(self, video_path):
        self.video_path = video_path

    def generate_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()