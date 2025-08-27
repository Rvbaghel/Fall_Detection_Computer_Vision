import cv2
import cvzone
import math
from ultralytics import YOLO

# Load your trained model (replace with the path to your best.pt file)
model = YOLO("best.pt")

# Load video
cap = cv2.VideoCapture("fall3.mp4")

# Load class names from classes.txt (should contain 'patient' and 'fall_pat')
classnames = []
with open("class.txt", "r") as f:
    classnames = f.read().splitlines()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (980, 740))

    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = int(box.cls[0])
            class_name = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            # Draw detection if confidence > 80%
            if conf > 80:
                cvzone.cornerRect(frame, [x1, y1, x2-x1, y2-y1], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_name} {conf}%', [x1 + 8, y1 - 12], thickness=2, scale=2)

                # Alert only if fall is detected
                if class_name == "fall_pat":
                    cvzone.putTextRect(frame, "FALL DETECTED!", [50, 50], thickness=3, scale=3, colorR=(0, 0, 255))

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("t"):
        break

cap.release()
cv2.destroyAllWindows()
