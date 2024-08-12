import os
import cv2  # Import the OpenCV library for computer vision tasks
import pickle
import numpy as np
import face_recognition
import cvzone


import numpy as np

from datetime import datetime
cap = cv2.VideoCapture(0)  # Create a VideoCapture object. The argument '1' specifies the camera index (0 is typically the default camera, and 1 is the second camera).
cap.set(3, 640)  # Set the width of the video capture to 640 pixels (3 is the property identifier for width).
cap.set(4, 480)  # Set the height of the video capture to 480 pixels (4 is the property identifier for height).

imgBackground = cv2.imread('Ressources/background.png')

# Importing the mode images into a list
folderModePath = 'Ressources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
#print(len(imgModeList))

# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
# print(studentIds)
print("Encode File Loaded")

while True:  # Start an infinite loop to continuously capture frames from the camera
    success, img = cap.read()  # Capture a frame from the video stream; 'success' indicates if the frame was captured successfully, and 'img' contains the image data

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)#this is to make the image smaller so that it does not take lot of time
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)#locate the face of the curr image
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)#encode the current face recognized

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[1]

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("matches", matches)
        print("faceDis", faceDis)

        matchIndex = np.argmin(faceDis)
        # print("Match Index", matchIndex)

        if matches[matchIndex]:
            # print("Known Face Detected")
            # print(studentIds[matchIndex])
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4# we have to multiply each by 4 because before we have reduced it to 0.25 meaning 1/4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)





    #cv2.imshow("WebCam", img)  # Display the captured image in a window titled "Face Attendance"
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)  # Wait for 1 millisecond for a key event; this allows the window to refresh and respond to user input

