import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, date


def imagespathnames(path):
    myList = os.listdir(path)
    imagesPath = []
    names = []
    for name in myList:
        currentImg = cv2.imread(f'{path}/{name}')
        imagesPath.append(currentImg)
        names.append(name.split(".")[0])
    return imagesPath, names


def findencodings(imagesPath):
    encodeList = []
    for img in imagesPath:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markattendace(name):
    path = "Attendance_Record"
    if not (os.path.exists(path)):
        os.mkdir(path)
    filename = date.today()
    filename = (f'{path}/{filename}.csv')
    if not (os.path.isfile(filename)):
        file = open(filename, "a")
        file.write("NAME,TIME")
        file.close()

    with open(filename, 'r+') as f:
        datelist = f.readlines()
        namelist = []
        lastcheckhour = ""
        for line in datelist:
            entry = line.split(',')
            namelist.append(entry[0])
            if (entry[0] == name):
                lastcheckhour = entry[1][0:2]
        if name not in namelist:
            now = datetime.now()
            datestring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{datestring}')
        else:
            now = datetime.now()
            now = now.strftime('%H:%M:%S')
            currenthour = now[0:2]
            if (lastcheckhour != currenthour):
                now = datetime.now()
                datestring = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{datestring}')
        f.close()


def main():
    path = 'images'
    imagespath, names = imagespathnames(path)
    encodings = findencodings(imagespath)

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        # reduce size of image
        imgsmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgsmall = cv2.cvtColor(imgsmall, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgsmall)
        encodeCurFrame = face_recognition.face_encodings(imgsmall, facesCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodings, encodeFace)
            faceDis = face_recognition.face_distance(encodings, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = names[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                # convert point to original from small
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                # create a box on detected face
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                # mark attendance
                markattendace(name)
        title = "Attendance by Face Detection"
        cv2.imshow(title, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print("Press Q for exit.")


if __name__ == "__main__":
    main()
