import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not found or not opened!")
    exit()
    
print("webcam found!Enter 'q' for exit..")

while True:
    ret , frame = cap.read()
    if not ret:
        print("Error!")
        break
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray , 1.3 , 5)
    
    for(x , y , w , h) in face:
        cv2.rectangle(frame , (x,y) , (x+y , y+h) , (255 , 0 ,0) , 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
    cv2.imshow('Face and Eye Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    