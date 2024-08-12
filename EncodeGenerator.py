import cv2

import face_recognition
import pickle
import os

# Importing student images
folderPath = 'Images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIds.append(os.path.splitext(path)[0])#to remove the .png we have to split it and then get the one at index 0

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