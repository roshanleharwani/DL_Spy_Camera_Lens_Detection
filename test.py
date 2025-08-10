import torch
import cv2
import numpy as np
from abc import ABC, abstractmethod
import time


class BaseDetector(ABC):
    """Abstract base class for all detectors"""

    def __init__(self, model_path, confidence_threshold=0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    @abstractmethod
    def load_model(self):
        """Load the YOLO model"""
        pass

    @abstractmethod
    def detect(self, frame):
        """Perform detection on the frame"""
        pass

    def preprocess_frame(self, frame):
        """Common preprocessing for YOLO models"""
        return frame


class ObjectDetector(BaseDetector):
    """First model: Detects phones and cameras"""

    def __init__(
        self, model_path, target_classes=["phone", "camera"], confidence_threshold=0.5
    ):
        self.target_classes = target_classes
        super().__init__(model_path, confidence_threshold)

    def load_model(self):
        """Load the object detection model"""
        try:
            self.model = torch.hub.load(
                "ultralytics/yolov5", "custom", path=self.model_path
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"Object detection model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading object detection model: {e}")

    def detect(self, frame):
        """Detect phones and cameras in the frame"""
        if self.model is None:
            return [], frame

        results = self.model(frame)
        detections = []
        annotated_frame = frame.copy()

        # Parse results
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf > self.confidence_threshold:
                class_name = self.model.names[int(cls)]
                if class_name.lower() in [tc.lower() for tc in self.target_classes]:
                    x1, y1, x2, y2 = map(int, box)
                    detections.append(
                        {
                            "class": class_name,
                            "confidence": conf,
                            "bbox": (x1, y1, x2, y2),
                            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                        }
                    )

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_frame,
                        f"{class_name}: {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        return detections, annotated_frame


class LensDetector(BaseDetector):
    """Second model: Detects camera lens when camera is found"""

    def __init__(self, model_path, confidence_threshold=0.5):
        super().__init__(model_path, confidence_threshold)

    def load_model(self):
        """Load the lens detection model"""
        try:
            self.model = torch.hub.load(
                "ultralytics/yolov5", "custom", path=self.model_path
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"Lens detection model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading lens detection model: {e}")

    def detect(self, frame, roi=None):
        """Detect camera lens in the frame or ROI"""
        if self.model is None:
            return [], frame

        # Use ROI if provided, otherwise use full frame
        detection_frame = frame[roi[1] : roi[3], roi[0] : roi[2]] if roi else frame

        results = self.model(detection_frame)
        detections = []
        annotated_frame = frame.copy()

        # Parse results
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf > self.confidence_threshold:
                class_name = self.model.names[int(cls)]
                x1, y1, x2, y2 = map(int, box)

                # Adjust coordinates if ROI was used
                if roi:
                    x1 += roi[0]
                    y1 += roi[1]
                    x2 += roi[0]
                    y2 += roi[1]

                detections.append(
                    {
                        "class": class_name,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    }
                )

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated_frame,
                    f"Lens: {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        return detections, annotated_frame


class IntelligentDetectionSystem:
    """Main system that manages the inheritance between models"""

    def __init__(
        self,
        object_model_path,
        lens_model_path,
        object_confidence=0.5,
        lens_confidence=0.5,
    ):
        # Initialize both detectors
        self.object_detector = ObjectDetector(
            object_model_path, confidence_threshold=object_confidence
        )
        self.lens_detector = LensDetector(
            lens_model_path, confidence_threshold=lens_confidence
        )

        # State management
        self.current_state = "OBJECT_DETECTION"  # or "LENS_DETECTION"
        self.tracked_objects = []
        self.last_detection_time = time.time()
        self.detection_timeout = 2.0  # seconds

        # Tracking parameters
        self.tracking_tolerance = 50  # pixels

    def is_object_still_present(self, current_detections):
        """Check if previously detected objects are still in frame"""
        if not self.tracked_objects or not current_detections:
            return False

        for tracked in self.tracked_objects:
            for current in current_detections:
                # Check if objects are close enough (simple distance tracking)
                tracked_center = tracked["center"]
                current_center = current["center"]
                distance = np.sqrt(
                    (tracked_center[0] - current_center[0]) ** 2
                    + (tracked_center[1] - current_center[1]) ** 2
                )

                if distance < self.tracking_tolerance:
                    return True
        return False

    def get_camera_roi(self, detections):
        """Get Region of Interest for camera objects"""
        camera_detections = [d for d in detections if d["class"].lower() == "camera"]
        if camera_detections:
            # Use the first camera detection as ROI
            bbox = camera_detections[0]["bbox"]
            # Expand ROI slightly for better lens detection
            margin = 20
            x1 = max(0, bbox[0] - margin)
            y1 = max(0, bbox[1] - margin)
            x2 = bbox[2] + margin
            y2 = bbox[3] + margin
            return (x1, y1, x2, y2)
        return None

    def process_frame(self, frame):
        """Main processing function that manages model switching"""
        current_time = time.time()

        if self.current_state == "OBJECT_DETECTION":
            # Use object detection model
            detections, annotated_frame = self.object_detector.detect(frame)

            if detections:
                # Objects detected, check if any are cameras
                camera_detected = any(
                    d["class"].lower() == "camera" for d in detections
                )

                if camera_detected:
                    # Switch to lens detection mode
                    self.current_state = "LENS_DETECTION"
                    self.tracked_objects = detections
                    self.last_detection_time = current_time
                    print("Switching to LENS_DETECTION mode")

                # Add state indicator to frame
                cv2.putText(
                    annotated_frame,
                    f"State: {self.current_state}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )

                return detections, annotated_frame

        elif self.current_state == "LENS_DETECTION":
            # First check if objects are still present
            object_detections, _ = self.object_detector.detect(frame)

            if self.is_object_still_present(object_detections):
                # Objects still present, use lens detector
                camera_roi = self.get_camera_roi(object_detections)
                lens_detections, annotated_frame = self.lens_detector.detect(
                    frame, camera_roi
                )

                # Also draw object detections
                for det in object_detections:
                    x1, y1, x2, y2 = det["bbox"]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_frame,
                        f"{det['class']}: {det['confidence']:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                self.last_detection_time = current_time

                # Add state indicator
                cv2.putText(
                    annotated_frame,
                    f"State: {self.current_state}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )

                return lens_detections + object_detections, annotated_frame

            else:
                # Objects gone, check timeout
                if current_time - self.last_detection_time > self.detection_timeout:
                    # Switch back to object detection
                    self.current_state = "OBJECT_DETECTION"
                    self.tracked_objects = []
                    print("Objects lost, switching back to OBJECT_DETECTION mode")

                # Continue with lens detection for now
                lens_detections, annotated_frame = self.lens_detector.detect(frame)
                cv2.putText(
                    annotated_frame,
                    f"State: {self.current_state} (timeout in {self.detection_timeout - (current_time - self.last_detection_time):.1f}s)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

                return lens_detections, annotated_frame

        # Default fallback
        return [], frame


# Usage example and main execution
def main():
    """Main function to run the intelligent detection system"""

    # Initialize the system with your model paths
    system = IntelligentDetectionSystem(
        object_model_path="best.pt",
        lens_model_path="runs/detect/train/weights/best.pt",
        object_confidence=0.5,
        lens_confidence=0.5,
    )

    # For webcam input
    cap = cv2.VideoCapture(0)

    # For video file input
    # cap = cv2.VideoCapture("path/to/your/video.mp4")

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    print("Starting intelligent detection system...")
    print("Press 'q' to quit, 's' to save current frame")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break

        # Process frame through the intelligent system
        detections, processed_frame = system.process_frame(frame)

        # Display results
        cv2.imshow("Intelligent Detection System", processed_frame)

        # Print detection info
        if detections:
            print(
                f"Frame {frame_count}: Detected {len(detections)} objects in {system.current_state} mode"
            )

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(f"detection_frame_{frame_count}.jpg", processed_frame)
            print(f"Saved frame {frame_count}")

        frame_count += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection system terminated")


if __name__ == "__main__":
    main()
