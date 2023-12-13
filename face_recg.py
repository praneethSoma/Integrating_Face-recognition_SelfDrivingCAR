import dlib
import cv2 as cv
import face_recognition
import sqlite3
import os
import numpy as np
import time

def initialize_db():
    if not os.path.exists("Users.sqlite"):
        conn = sqlite3.connect("Users.sqlite")
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                encoding BLOB NOT NULL,
                is_master BOOLEAN DEFAULT FALSE,
                phone_number TEXT
            );
        ''')
        conn.commit()
    else:
        conn = sqlite3.connect("Users.sqlite")
    return conn

def master_exists(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE is_master = 1")
    data = cursor.fetchone()
    return data is not None

def master_authenticate_timer(conn):
    print("Master user must authenticate within 15 seconds to allow registration.")
    for i in range(15, 0, -1):
        print(f"Time left: {i} seconds")
        time.sleep(1)
        frame = capture_frame()
        encoding = detect_encode_face(frame)
        if encoding is not None and authenticate_master_user(encoding, conn):
            print("Master user authenticated!")
            return True
    return False

def capture_frame():
    cap = cv.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame

#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def detect_encode_face(frame):
    face_encodings = face_recognition.face_encodings(frame)

    if face_encodings:
        return face_encodings[0]
    else:
        return None

def register_master_user(name, encoding, conn,phone_number):
    cursor = conn.cursor()
    encoded_data = encoding.tobytes()
    cursor.execute("INSERT INTO users (name, encoding, is_master,phone_number) VALUES (?, ?, ?,?)", (name, encoded_data, True,phone_number))
    conn.commit()
    print("Master user registered successfully.")

def register_driver(name, encoding, conn,phone_number):
    cursor = conn.cursor()
    encoded_data = encoding.tobytes()
    cursor.execute("INSERT INTO users (name, encoding, is_master,phone_number) VALUES (?, ?, ?,?)", (name, encoded_data, False,phone_number))
    conn.commit()
    print("Driver registered successfully.")

def authenticate_master(encoding, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT encoding FROM users WHERE is_master = ?", (True,))
    master_encoding_data = cursor.fetchone()
    if master_encoding_data:
        master_encoding = np.frombuffer(master_encoding_data[0], dtype=np.float64)
        matches = face_recognition.compare_faces([master_encoding], encoding)
        if matches[0]:
            return True
    return False

def authenticate_driver(encoding, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT encoding FROM users")
    stored_encodings_data = cursor.fetchall()
    stored_encodings = [np.frombuffer(row[0], dtype=np.float64) for row in stored_encodings_data]
    matches = face_recognition.compare_faces(stored_encodings, encoding)
    if True in matches:
        return True
    return False

def capture_and_save_image(username):
    frame = capture_frame()
    if not os.path.exists("captured_images"):
        os.makedirs("captured_images")
    cv.imwrite(f"captured_images/{username}.jpg", frame)
    return frame

def main():
    conn = initialize_database()

    print("Do you want to (1) Register as master user, (2) Register as a driver, or (3) Authenticate?")
    choice = int(input())

    if choice == 1:
        if master_exists(conn):
            print("Master user already exists!")
            return
        print("Enter name for master user:")
        name = input().strip()
        frame = capture_and_save_image(name)
        encoding = detect_encode_face(frame)
        if encoding is not None:
            register_master_user(name, encoding, conn)
            print("Face detected.")
        else:
            print("Face not detected. Please try again.")

    elif choice == 2:
        if not master_authenticate_timer(conn):
            print("Master user authentication failed. Cannot register a new driver.")
            return
        print("Enter name for driver:")
        name = input().strip()
        frame = capture_and_save_image(name)
        encoding = detect_encode_face(frame)
        if encoding is not None:
            # If the new user is the same as the master user, then no need to register
            if authenticate_master_user(encoding, conn):
                print("This face is recognized as the master user. No need to register again.")
                return
            register_driver(name, encoding, conn)
        else:
            print("Face not detected. Please try again.")

    elif choice == 3:
        frame = capture_frame()
        encoding = detect_encode_face(frame)
        if encoding is not None:
            driver_authenticated = authenticate_driver(encoding, conn)
            if driver_authenticated:
                print("Driver authenticated.")
            else:
                print("Authentication failed.")
        else:
            print("Face not detected. Please try again.")

    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
