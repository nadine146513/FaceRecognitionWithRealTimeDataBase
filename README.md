# Real-Time Face Attendance System

A real-time face recognition-based attendance system that uses a camera to capture and verify faces against a stored database. When a recognized face is detected, the system marks the attendance and updates the record in Firebase.

## Table of Contents
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Screenshots

<div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
    <img src="https://github.com/user-attachments/assets/e93e7fae-730b-483e-92f5-b2c2aa0ea587" alt="Screenshot 2024-08-26 132355" style="width: 30%; margin: 10px;"/>
    <img src="https://github.com/user-attachments/assets/b97b38c6-7c04-4f35-8e58-e272ab7c3ab6" alt="Screenshot 2024-08-26 132433" style="width: 30%; margin: 10px;"/>
    <img src="https://github.com/user-attachments/assets/e0f52ad2-969e-407d-9fb3-dce891310944" alt="Screenshot 2024-08-26 135523" style="width: 30%; margin: 10px;"/>
    
</div>

## Installation

### Requirements

- Python 3.x
- OpenCV (`opencv-python` package)
- face_recognition library
- Firebase Admin SDK
- Firebase Realtime Database
- Firebase Storage
- `cvzone` library

### Steps to Install

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Face-Attendance-Realtime.git
---

2. **Navigate to the project folder:**

   ```bash
   cd Face-Attendance-Realtime
---

 3. **Install required dependencies:**

    ```bash
    pip install -r requirements.txt
---

## Setup Firebase:

1. Create a Firebase project at [Firebase Console](https://console.firebase.google.com/).
2. Add Firebase Realtime Database and Firebase Storage to your project.
3. Download the `serviceAccountKey.json` file and place it in the project directory.

## Usage

### Prepare your dataset:

1. Add student images to the **Images** directory.
2. Ensure the image names are the student IDs (e.g., `321654.png`).

### Run the face encoding script to store the facial encodings in a file:

    ```bash
    python face_encoding.py
  ---

## Start the attendance system:

### Run the main script:

    ```bash
    python main.py
  ---
Attendances will be marked automatically once a recognized face is detected

## Firebase Data Structure

The attendance data will be stored in Firebase Realtime Database under the **Students** node. Each student record will contain:

    ```json
    "studentId": {
      "name": "Student Name",
      "major": "Student Major",
      "starting_year": 2021,
      "total_attendance": 5,
      "standing": "A",
      "year": 2,
      "last_attendance_time": "2022-12-11 00:54:34"
    }
  ---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
