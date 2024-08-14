import cv2

import face_recognition
import pickle
import os
import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import  storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendancerealtime-8fd50-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendancerealtime-8fd50.appspot.com"
})



# Importing student images
folderPath = 'Images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIds.append(os.path.splitext(path)[0])#to remove the .png we have to split it and then get the one at index 0
    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)
print(studentIds)


def findEncodings(imagesList):
    # Create an empty list to store the encodings of each image
    encodeList = []
    # Iterate through each image in the list
    for img in imagesList:
        # Convert the image from BGR (OpenCV's default) to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Find the encodings for all faces in the image (assuming only one face per image)
        encode = face_recognition.face_encodings(img)[0]
        # Append the encoding to the list
        encodeList.append(encode)
    # Return the list of encodings
    return encodeList

print("Encoding Started ...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")