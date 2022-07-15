import streamlit as st
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import cvlib as cv
from mtcnn import MTCNN
import pickle
# if __name__ == "__main__":
#     st.title("Emotion Detection App")
#     st.image("/Users/ojonyeagwu/Desktop/emotion/angry.jpeg")

# Here is the function for UI

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
classNames = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprise']
model = tf.keras.models.load_model("face_emotion_rec_v2.h5")  

def prediction_img(img,detect_model,classNames):
    #frame = cv2.imread(img)
    frame = Image.open(img)
    frame = np.array(frame)
    face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray_img,1.1,4)
    for x,y,w,h in faces:
        roi_gray_img = gray_img[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        facess = face_detect.detectMultiScale(roi_gray_img)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for (ex,ey,ew,eh) in facess:
                face_roi = roi_color[ey: ey+eh,ex:ex +ew]    

    final_img = cv2.resize(face_roi,(224,224))
    final_img = np.expand_dims(final_img,axis=0) # need 4th dimension
    final_img = final_img/255 # normalizing

    prediction = detect_model.predict(final_img)
    pred = np.argmax(prediction[0])
    cv2.putText(final_img,classNames[pred], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    answer = classNames[pred]
    st.write(answer)
    #return classNames[pred]


def main():
    st.title("Emotion detection using Tensorflow")
    st.write("Use side bar to select what type of detection")
    st.sidebar.write("Select an Option Below")

    activities = [
                  "Image Detection", "Live Video Feed Detection"]
    #choice = st.sidebar.selectbox("select an option", activities)
    choice = st.sidebar.selectbox("",activities)

    if choice == "Image Detection":
        st.title("Image Detection")
        uploaded_file = st.file_uploader(
            "Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

        if uploaded_file is not None:
            image_file = uploaded_file.getvalue()
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(' ')

            with col2:
                st.image(image_file, width=500)

            with col3:
                st.write(' ')
            
            if st.button("Process"):
                #FRAME_WINDOW = st.image([])
                prediction_img(uploaded_file,model,classNames)


    if choice ==  "Live Video Feed Detection":
        st.header("Webcam Live Feed")

        run = st.checkbox('Run')
        vid_obj = cv2.VideoCapture(0)
        frame = st.empty() 
        #success = True

        while run:
            run, image = vid_obj.read()
            if not run:
                continue
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
            face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces = face_haar_cascade.detectMultiScale(gray_img)

            for x,y,w,h in faces:
                cv2.rectangle(image, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  2)
                roi_gray = gray_img[y-5:y+h+5,x-5:x+w+5]
                roi_gray=cv2.resize(roi_gray,(224,224))
                image_pixels = img_to_array(roi_gray)
                image_pixels = np.expand_dims(image_pixels, axis = 0)
                image_pixels /= 255
                predictions = model.predict(image_pixels)
                max_index = np.argmax(predictions[0])
                emotion_prediction = classNames[max_index]
                
            #     cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            #     if len(faces) == 0:
            #         print("Face not detected")
            #     else:
            #         for (ex,ey,ew,eh) in faces:
            #             face_roi = roi_color[ey: ey+eh,ex:ex +ew]       
            # if face_roi is not None:
            #         final_img = cv2.resize(face_roi,(224,224))
            #         final_img = np.expand_dims(final_img,axis=0) # need 4th dimension
            #         final_img = final_img/255 # normalizing       
            #         prediction = model.predict(final_img)
            #         pred = np.argmax(prediction[0])

            #         x1,y1,w1,h1 = 0,0,175,75
                cv2.putText(image,emotion_prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame.image(image, channels="BGR") 




        # st.header("Webcam Live Feed")
        # run = st.checkbox('Run')
        # FRAME_WINDOW = st.image([])
        
        
        # while run:
        #     webcam = cv2.VideoCapture(0)
        #      # read frame from webcam 
        #     status, frame = webcam.read()

        #     # apply face detection
        #     face, confidence = cv.detect_face(frame)

        #     # loop through detected faces
        #     for idx, f in enumerate(face):

        #         # get corner points of face rectangle        
        #         (startX, startY) = f[0], f[1]
        #         (endX, endY) = f[2], f[3]

        #         # draw rectangle over face
        #         cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        #         # crop the detected face region
        #         face_crop = np.copy(frame[startY:endY,startX:endX])

        #         if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
        #             continue

        #         # preprocessing for gender detection model
        #         face_crop = cv2.resize(face_crop, (224,224))
        #         face_crop = face_crop.astype("float") / 255.0
        #         face_crop = img_to_array(face_crop)
        #         face_crop = np.expand_dims(face_crop, axis=0)

        #         # apply gender detection on face
        #         conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        #         # get label with max accuracy
        #         idx = np.argmax(conf)
        #         label = classNames[idx]

        #         label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        #         Y = startY - 10 if startY - 10 > 10 else startY + 10

        #         # write label and confidence above face rectangle
        #         cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
        #                     0.7, (0, 255, 0), 2)
        #         FRAME_WINDOW.image(frame)    


        
        # st.header("Webcam Live Feed")
        # run = st.checkbox('Run')
        # face_roi = None   
        # cap = cv2.VideoCapture(0)
        # while(cap.isOpened()):

        #     while True:
        #         ret, frame = cap.read()
        #         if not ret:
        #             continue
        #         gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #         faces = face_cascade.detectMultiScale(gray_img,1.1,4)

        #         for x,y,w,h in faces:
        #                 roi_gray_img = gray_img[y:y+h,x:x+w]
        #                 roi_color = frame[y:y+h,x:x+w]
        #                 cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #                 facess = face_cascade.detectMultiScale(roi_gray_img)
        #                 if len(facess) == 0:
        #                     print("Face not detected")
        #                 else:
        #                     for (ex,ey,ew,eh) in facess:
        #                         face_roi = roi_color[ey: ey+eh,ex:ex +ew]       
        #         if face_roi is not None:
        #                 final_img = cv2.resize(face_roi,(224,224))
        #                 final_img = np.expand_dims(final_img,axis=0) # need 4th dimension
        #                 final_img = final_img/255 # normalizing       

        #                 model = tf.keras.models.load_model("face_emotion_rec_v2.h5") 
        #                 prediction = model.predict(final_img)
        #                 pred = np.argmax(prediction[0])

        #                 x1,y1,w1,h1 = 0,0,175,75

        #                 #cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0,),-1)

        #                 #cv2.putText(frame,classNames[pred],(x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        #                 cv2.putText(frame,classNames[pred], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #                 #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
        #         cv2.imshow("Face emotion recognation", frame)
        #         if cv2.waitKey(1) & 0xFF==ord('q'): 
        #             break
            
        #     cap.release()
        #     cv2.destroyAllWindows()
        else:
            # webcam.release()
            # cv2.destroyAllWindows
            st.write('Webcam Feed stopped')


if __name__ == "__main__":
    main()
