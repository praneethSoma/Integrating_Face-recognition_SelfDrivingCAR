from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
                             QLabel, QInputDialog, QMessageBox, QDialog, QProgressBar, QComboBox,QDesktopWidget)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon
import sqlite3
import os
import numpy as np
import time
from face_recg import (authenticate_master, authenticate_driver, initialize_db, capture_frame, 
                       detect_encode_face, master_exists, register_master_user, register_driver)
                       
from Modified_GPS_System import select_spawn_and_destination
from Modified_Automatic_Control_Refined import  World
import subprocess
import clx.xms 
import requests 
import warnings
import face_recognition
warnings.simplefilter("ignore", DeprecationWarning)
from twilio.rest import Client
#import alert_system
import pickle
#import clx.xms


                       
class GPSDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GPS Module")
        self.setGeometry(500, 500, 300, 150)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        self.spawn_label = QLabel("Select Spawn Point:")
        layout.addWidget(self.spawn_label)

        self.spawn_combobox = QComboBox()
        # TODO: Populate the combobox with actual spawn points
        self.spawn_combobox.addItem("Spawn Point 1")
        self.spawn_combobox.addItem("Spawn Point 2")
        layout.addWidget(self.spawn_combobox)

        self.destination_label = QLabel("Select Destination:")
        layout.addWidget(self.destination_label)

        self.destination_combobox = QComboBox()
        # TODO: Populate the combobox with actual destinations
        self.destination_combobox.addItem("Destination 1")
        self.destination_combobox.addItem("Destination 2")
        layout.addWidget(self.destination_combobox)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.on_start_clicked)
        layout.addWidget(self.start_btn)

        self.setLayout(layout)

    def on_start_clicked(self):
        # TODO: Implement the logic to start the car using the selected spawn and destination points
        spawn_point = self.spawn_combobox.currentText()
        destination = self.destination_combobox.currentText()
        print(f"Starting car from {spawn_point} to {destination}")
        self.accept()                       
   

   
class AddUserDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add User")
        self.setGeometry(500, 500, 300, 150)
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.remaining_time = 15  # 15 seconds countdown

        self.label = QLabel(f"Authenticate Master User\n\n{self.remaining_time} seconds remaining")
        self.layout.addWidget(self.label)

        self.progress = QProgressBar(self)
        self.progress.setMaximum(15)
        self.progress.setValue(15)
        self.layout.addWidget(self.progress)

        self.setLayout(self.layout)

        self.conn = initialize_db()  # Initialize the database and get the connection

        self.timer.start(1000)  # 1 second interval

    def update_timer(self):
        self.remaining_time -= 1
        self.label.setText(f"Authenticate Master User\n\n{self.remaining_time} seconds remaining")
        self.progress.setValue(self.remaining_time)
        
        self.authenticate_master_user()  # Attempt authentication during each timer tick

        if self.remaining_time <= 0:
            self.timer.stop()
            self.reject()  # Close the dialog if time runs out

    def authenticate_master_user(self):
        frame = capture_frame()  # Capture the current frame
        encoding = detect_encode_face(frame)  # Detect and encode the face from the frame

        if encoding is not None:
            is_master_authenticated = authenticate_master(encoding, self.conn)
            if is_master_authenticated:
                self.timer.stop()
                self.accept()  # Close the dialog when authentication is successful
                
            else:
                self.send_alert() 
                
   
                
    def on_add_user_clicked(self):
       
            
        # Step 1 and 2: Open the AddUserDialog for master authentication
        dialog = AddUserDialog(self)
        result = dialog.exec_()
        self.setGeometry(400, 400, 400, 300)

        # Main Layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        # Step 4: Check if master user is authenticated
        if result == QDialog.Accepted:
            # Step 5: Ask for the new user's details
            name, ok1 = QInputDialog.getText(self, 'Input Dialog', 'Enter new user name:')
            phone_number, ok2 = QInputDialog.getText(self, 'Input Dialog', 'Enter phone number:')
            
            if ok1 and ok2:
                # Step 6: Capture the new user's face and compare with master user's encoding
                conn = initialize_db()
                frame = capture_frame()
                encoding = detect_encode_face(frame)
                
                # Check if the new user is the same as the master user
                if authenticate_master(encoding, conn):
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("The new user is the same as the master user. Master user already authenticated.")
                    msg.setWindowTitle("Warning")
                    QTimer.singleShot(2000, msg.close)
                    msg.exec_()
                    conn.close()
                    return
                
                if encoding is not None:
                    register_driver(name, encoding, conn, phone_number)
                    conn.close()
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setText(f"User {name} added successfully!")
                    msg.setWindowTitle("Success")
                    msg.exec_()
                else:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Face not detected. Please try again.")
                    msg.setWindowTitle("Warning")
                    QTimer.singleShot(2000, msg.close)
                    msg.exec_()
                    conn.close()
        else:
            # Master authentication failed or time ran out
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Master user authentication failed!")
            msg.setWindowTitle("Error")
            QTimer.singleShot(2000, msg.close)
            msg.exec_()


    def closeEvent(self, event):
        # Close the database connection when the dialog is closed
        self.conn.close()
        event.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window setup
        self.setWindowTitle("Self-Driving Car Controller")
        self.setGeometry(400, 400, 1280,720)

        # Main Layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Master User Button
        self.master_user_btn = QPushButton("Master User")
        self.master_user_btn.clicked.connect(self.on_master_user_clicked)

        # Add User Button
        self.add_user_btn = QPushButton("Add User")
        self.add_user_btn.clicked.connect(self.on_add_user_clicked)

        # Start Button
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.on_start_clicked)

        # Add buttons to the layout
        layout.addWidget(self.master_user_btn)
        layout.addWidget(self.add_user_btn)
        layout.addWidget(self.start_btn)

        # Setting the main layout
        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        
        # Set up a timer for periodic face recognition
        self.face_recognition_timer = QTimer(self)
        self.face_recognition_timer.timeout.connect(self.perform_periodic_face_recognition)
        # Set the timer to trigger every 30 seconds (30000 milliseconds)
        self.face_recognition_timer.start(500000)
        self.carla_process = None
        self.authenticated_user = None
        self.master_phone_number = None
        self.alert_process=None



    def on_master_user_clicked(self):
            conn = initialize_db()
            if master_exists(conn):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("A master user already exists!")
                msg.setWindowTitle("Info")
                QTimer.singleShot(2000, msg.close)
                msg.exec_()
                conn.close()
                return

            name, ok1 = QInputDialog.getText(self, 'Input Dialog', 'Enter master user name:')
            phone_number, ok2 = QInputDialog.getText(self, 'Input Dialog', 'Enter phone number:')
            if ok1 and ok2:
                frame = capture_frame()
                encoding = detect_encode_face(frame)
                if encoding is not None:
                    register_master_user(name, encoding, conn, phone_number)
                    conn.close()
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setText(f"Master user {name} registered successfully!")
                    msg.setWindowTitle("Success")
                    msg.exec_()
                else:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Face not detected. Please try again.")
                    msg.setWindowTitle("Warning")
                    QTimer.singleShot(2000, msg.close)
                    msg.exec_()
                    conn.close()

    def on_add_user_clicked(self):
        # Step 1 and 2: Open the AddUserDialog for master authentication
        dialog = AddUserDialog(self)
        result = dialog.exec_()
        
        # Step 4: Check if master user is authenticated
        if result == QDialog.Accepted:
            # Step 5: Ask for the new user's details
            name, ok1 = QInputDialog.getText(self, 'Input Dialog', 'Enter new user name:')
            phone_number, ok2 = QInputDialog.getText(self, 'Input Dialog', 'Enter phone number:')
            
            if ok1 and ok2:
                # Step 6: Capture the new user's face and store the details in the database
                conn = initialize_db()
                frame = capture_frame()
                encoding = detect_encode_face(frame)
                if encoding is not None:
                    register_driver(name, encoding, conn, phone_number)
                    conn.close()
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setText(f"User {name} added successfully!")
                    msg.setWindowTitle("Success")
                    msg.exec_()
                else:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Face not detected. Please try again.")
                    msg.setWindowTitle("Warning")
                    msg.exec_()
                    conn.close()
        else:
            # Master authentication failed or time ran out
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Master user authentication failed!")
            msg.setWindowTitle("Error")
            QTimer.singleShot(2000, msg.close)
            msg.exec_()
            
            
    def get_master_user_phone(self, conn):
        cursor = conn.cursor()
        cursor.execute("SELECT phone_number FROM users WHERE is_master = 1")
        master_user_data = cursor.fetchone()
        if master_user_data:
            return master_user_data[0]
        else:
            return None
            
    
    def on_start_clicked(self):
        conn = initialize_db()
        frame = capture_frame()
        encoding = detect_encode_face(frame)
        self.master_phone_number = self.get_master_user_phone(conn)
        if encoding is not None:
            if authenticate_master(encoding, conn) or authenticate_driver(encoding, conn):
                self.initial_user_encoding = encoding
                #self.master_phone_number = self.get_master_user_phone(conn) 
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("User authenticated.")
                msg.setWindowTitle("Success")
                QTimer.singleShot(2000, msg.close)
                msg.exec_()

                # Get the selected spawn and destination points using the GPS module
                spawn_term, spawn, dest_term, dest = select_spawn_and_destination()
                print(f"Chosen Spawn: {spawn_term} at {spawn}")
                print(f"Chosen Destination: {dest_term} at {dest}")

                # Start the CARLA process
                self.carla_process = subprocess.Popen(["python", "Automatic_Control.py"])

                # Start the periodic face recognition timer
                self.face_recognition_timer.start(500000)

            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Authentication failed.")
                msg.setWindowTitle("Failed")
                QTimer.singleShot(2000, msg.close)
                self.alert_process = subprocess.Popen(["python", "alert_system.py"])
                msg.exec_()
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Face not detected. Please try again.")
            msg.setWindowTitle("Warning")
            QTimer.singleShot(2000, msg.close)
            self.alert_process = subprocess.Popen(["python", "alert_system.py"])
            msg.exec_()
        conn.close()

          
    def perform_periodic_face_recognition(self):
    
        if self.carla_process is None:  # If the CARLA process hasn't started yet
            return

        # Check if CARLA process is still active
        if self.carla_process.poll() is not None:  # If the CARLA process has terminated
            self.face_recognition_timer.stop()  # Stop the periodic face recognition
            return

        conn = initialize_db()
        frame = capture_frame()
        current_encoding = detect_encode_face(frame)

        if current_encoding is None:
            self.show_warning_message("Face not detected during periodic check. Please ensure visibility.")
            self.stop_carla_process()
            self.alert_process = subprocess.Popen(["python", "alert_system.py"])
            
        elif not self.is_same_user(current_encoding, self.initial_user_encoding):
            self.show_warning_message("Unauthorized driver detected. Stopping the vehicle.")
            self.stop_carla_process()
            self.alert_process = subprocess.Popen(["python", "alert_system.py"])
            

        
        conn.close()  
        
    def is_same_user(self, current_encoding, initial_encoding):
        # Compare the current face encoding with the initial encoding
        # A lower tolerance makes the comparison stricter
        if current_encoding is None or initial_encoding is None:
            return False
        return face_recognition.compare_faces([initial_encoding], current_encoding, tolerance=0.5)[0]
        
    def show_warning_message(self, text):
        # This function displays a warning message box with the given text.
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText(text)
        msg.setWindowTitle("Warning")
        msg.setStandardButtons(QMessageBox.Ok)
        QTimer.singleShot(2000, msg.close)
        msg.exec_()

    def stop_carla_process(self):
        # This function stops the CARLA process and the periodic face recognition timer.
        if self.carla_process:
            self.carla_process.terminate()
            # Optionally wait for the process to terminate
            self.carla_process.wait()
            self.carla_process = None
        self.face_recognition_timer.stop()


# Running the application
if __name__ == "__main__":
    app = QApplication([])
# Load the stylesheet
    with open("stylesheet.qss", "r") as file:
        stylesheet = file.read()
    app.setStyleSheet(stylesheet)
    
    window = MainWindow()
    window.show()
    app.exec_()
