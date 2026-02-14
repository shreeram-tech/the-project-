import cv2
from ultralytics import YOLO
import torch

# 1. Force the model to use your RTX 5060 (CUDA)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- DUM-E Vision System: Using {device} ---")

# 2. Load the model (Fastest version for real-time)
model = YOLO("yolov8m.pt").to(device)

# 3. Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 4. Run Detection
    results = model(frame, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            # Get Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point for the robot
            
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            # 5. Output for DUM-E's Brain
            if conf > 0.5:
                print(f"Target: {label} | Center: ({cx}, {cy})")

                # Visuals
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"{label}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("DUM-E Vision", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
#press q to quit the program
