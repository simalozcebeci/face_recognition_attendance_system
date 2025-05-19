import cv2
import pickle
import face_recognition
from ultralytics import YOLO
from datetime import datetime
import os
import csv
import socket

def send_tcp_to_esp32(result):
    try:
        esp32_ip = "192.168.1.150"  # BURAYI kendi ESP32 IP adresinle değiştir
        port = 1234  # ESP32'nin dinlediği port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((esp32_ip, port))
            a = s.sendall(b'1' if result else b'0')
            print("Information sent to TCP")
    except Exception as e:
        print(f"[ERROR] TCP failed: {e}")


# Load YOLOv8 face detection model
model = YOLO("yolov8n-face.pt")

# Load known face encodings
with open("known_faces.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

# CSV file for attendance
CSV_PATH = "attendance.csv"
recognized_set = set()

# Always reset CSV file at the start
with open(CSV_PATH, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "timestamp"])


# Start webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_img = frame[y1:y2, x1:x2]

        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_face)

        if encodings:
            matches = face_recognition.compare_faces(known_encodings, encodings[0], tolerance=0.5)
            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]

                if name not in recognized_set:
                    recognized_set.add(name)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    with open(CSV_PATH, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([name, timestamp])
                    print(f"[✓] {name} recognized and added to attendance.")
                    send_tcp_to_esp32(result=True) 
                    print(f"esp1")

                # Draw green box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # Draw red box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                send_tcp_to_esp32(result=False) 
                print(f"esp0.")

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Camera stopped.")
