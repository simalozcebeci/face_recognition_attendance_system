import face_recognition
import os
import pickle

ENCODINGS_PATH = "known_faces.pkl"
KNOWN_FACES_DIR = "known_faces"

known_encodings = []
known_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])  # "ahmet.jpg" → "ahmet"

with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print(f"{len(known_names)} kişi encode edildi ve kayit edildi.")
