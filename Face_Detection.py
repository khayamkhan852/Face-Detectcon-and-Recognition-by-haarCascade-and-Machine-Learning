import cv2
import pickle
#Captured = cv2.VideoCapture(0,cv2.CAP_DSHOW)
Captured = cv2.VideoCapture('khayam.mp4')


Face_Cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

label = {}
with open("labels.pickle", "rb") as f:
    og_label = pickle.load(f)
    label = {v:k for k,v in og_label.items()}

Captured.set(cv2.CAP_PROP_FRAME_WIDTH,640);
Captured.set(cv2.CAP_PROP_FRAME_HEIGHT,480);

while True:
    returned, frame = Captured.read()
    frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
    Gray_Frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    Faces = Face_Cascade.detectMultiScale(Gray_Frame, scaleFactor = 1.5, minNeighbors = 5)


    for (x, y , w, h) in Faces:
        #print(x,y,w,h)
        #Lets draw rectangle
        roi_gray = Gray_Frame[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >=45:# and conf <=85:
            #print(conf)
            #print(label[id_])
            Font = cv2.FONT_HERSHEY_SIMPLEX
            name = label[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x,y),Font,1,color,stroke,cv2.LINE_AA)


        #img_item = "7.png"
        #cv2.imwrite(img_item,roi_color)
        color = (255, 0, 0) #BGR

        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

    cv2.imshow('Face Detection', frame)
    # press 'q' on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

Captured.release()
cv2.destroyAllWindows()
