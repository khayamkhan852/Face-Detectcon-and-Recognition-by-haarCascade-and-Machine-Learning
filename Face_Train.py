import os
import cv2
from PIL import Image
import numpy as np
import pickle

Face_Cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

#opening my image directory
current_id = 0
label_ids = {} #empty directory
x_train = []
y_labels = []
 #where ever this file is saved i am looking for that path
#it will print C:\Users\khaya\Desktop\face detectcon haarCascade
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#now opening image directory
Image_Dir = os.path.join(BASE_DIR,'images') #give me the images


#now i wanna see the images
for root, dirs, files in os.walk(Image_Dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()
            #print(label,path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            #print(label_ids)
            Pil_Image = Image.open(path).convert('L') #converting into grayscale
            size = (550,550)
            final_image = Pil_Image.resize(size,Image.ANTIALIAS)
            Image_Array = np.array(final_image,'uint8')
            #print(Image_Array)
            Faces = Face_Cascade.detectMultiScale(Image_Array, scaleFactor = 1.5, minNeighbors = 5)


            for (x,y,w,h) in Faces:
                roi = Image_Array[y:y+h , x:x+w]
                x_train.append(roi)
                y_labels.append(id_)


#labels from directories
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainer.yml')
