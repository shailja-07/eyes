from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io

app = FastAPI()


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

@app.post("/detect-face-and-eyes/")
async def detect_face_and_eyes(file: UploadFile = File(...)):
    try:
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]

            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            eyes = sorted(eyes, key=lambda e: e[0]) 

            if len(eyes) >= 2:
                (ex1, ey1, ew1, eh1) = eyes[0] 
                left_eye = roi_color[ey1:ey1+eh1, ex1:ex1+ew1]
                enlarged_left_eye = cv2.resize(left_eye, (2*ew1, 2*eh1))  

                (ex2, ey2, ew2, eh2) = eyes[1] 
                right_eye = roi_color[ey2:ey2+eh2, ex2:ex2+ew2]
                enlarged_right_eye = cv2.resize(right_eye, (2*ew2, 2*eh2))  

               
                roi_color[ey1:ey1+eh1, ex1:ex1+ew1] = enlarged_left_eye
                roi_color[ey2:ey2+eh2, ex2:ex2+ew2] = enlarged_right_eye

                
                _, img_bytes = cv2.imencode('.jpg', image)
                img_io = io.BytesIO(img_bytes.tobytes())
                return StreamingResponse(img_io, media_type="image/jpeg")

        return {"message": "No face or eyes detected."}

    except Exception as e:
        return {"error": str(e)}

