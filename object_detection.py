import cv2
import time
import numpy as np
import logging
from ultralytics import YOLO

# Initialize logging
logging.basicConfig(filename='detection_metrics.log', level=logging.INFO, format='%(asctime)s - %(message)s', force=True)

# Load the YOLOv11 model
model = YOLO('yolo11s.pt')

# Open the webcam stream
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Dictionary to track object positions for trajectory estimation
object_tracker = {}
prev_time = None  # For tracking FPS without ROI

try:
    while True:
        start_time = time.time()  # Start time for FPS with ROI tracking

        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Perform object detection
        results = model.predict(frame, imgsz=640)
        detections = results[0].boxes

        # Log the total number of objects detected in this frame
        logging.info(f"Total objects detected: {len(detections)}")

        for i, box in enumerate(detections):

            # Extract bounding box coordinates and detection details
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Log detection details (bounding box and confidence score)
            logging.info(f"Detected: {class_name} | Confidence: {conf:.2f} | Box: ({x1}, {y1}, {x2}, {y2})")

            # ROI Optimization: Log the coordinates and size of the ROI
            roi = frame[y1:y2, x1:x2]
            logging.info(f"ROI Coordinates: ({x1}, {y1}, {x2}, {y2}) | ROI Size: {roi.shape}")

            # Trajectory Estimation: Track object movement and predict future positions
            if i in object_tracker:
                prev_center = object_tracker[i]
                velocity = np.array(center) - np.array(prev_center)
                predicted_position = np.array(center) + velocity

                # Draw trajectory line
                cv2.line(frame, tuple(center), tuple(predicted_position), (0, 0, 255), 2)

                # Log velocity and predicted position
                logging.info(f"Object {i} | Velocity: {velocity} | Predicted Position: {predicted_position}")

            # Update object tracker
            object_tracker[i] = center

        # Calculate FPS with ROI and log it
        fps_roi = 1 / (time.time() - start_time)
        logging.info(f"FPS with ROI: {fps_roi:.2f}")

        # Calculate FPS without ROI if this is not the first frame
        if prev_time is not None:
            fps_no_roi = 1 / (time.time() - prev_time)
            logging.info(f"FPS without ROI: {fps_no_roi:.2f}")

        # Update previous time for the next frame's calculation
        prev_time = time.time()

        # Display the frame
        cv2.imshow("Final System Integration", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:

    # Log any errors encountered during execution
    logging.error(f"Error encountered: {str(e)}")

finally:
    
    # Ensure resources are released properly
    cap.release()
    cv2.destroyAllWindows()
