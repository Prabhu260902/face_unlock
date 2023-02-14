import cv2
import numpy as np
import os
import pickle
from os.path import isfile, join
from os import listdir



downloadsFolder = r'C:\\Users\\Prabhu\\Desktop\\unlock\\faces\\'
model = cv2.face.LBPHFaceRecognizer_create()


if not os.listdir(downloadsFolder):
    face_classifier = cv2.CascadeClassifier('C:\\Users\\Prabhu\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

    def face_extractor(img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        if faces is ():
            return None
        
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h, x:x+w]

        return cropped_face

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = downloadsFolder + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)

            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Cropper', face)
        

        if cv2.waitKey(1) == 13 or count == 100:
            break
            
    cap.release()
    cv2.destroyAllWindows()      
    print("Collecting Samples Complete")





    
    onlyfiles = [f for f in listdir(downloadsFolder) if isfile(join(downloadsFolder, f))]

    Training_Data, Labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = downloadsFolder + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)


    with open("labels.pickle", "wb") as f:
        pickle.dump(np.asarray(Labels), f)

    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print("Model trained sucessefully")

    model.save("trained-model.yml")



model.read("trained-model.yml")




face_classifier = cv2.CascadeClassifier('C:\\Users\\Prabhu\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        results = model.predict(face)
        
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is User'
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence >= 90:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            break;
        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )

    except:
        cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13:
        break
        
cap.release()
cv2.destroyAllWindows() 
