import cv2
import streamlit as st
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras. preprocessing import image
font = cv2.FONT_HERSHEY_SIMPLEX

   
org = (0, 30)

       # fontScale
fontScale = 0.5

       # Blue color in BGR
color = (255, 0, 0)

       # Line thickness of 2 px
thickness = 2
   
st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
FRAME_WINDOW1= st.image([])
camera = cv2.VideoCapture(0)
model2 = tf.keras.models.load_model('model1.h5')


while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    ret, frame = camera .read()
    time.sleep(0.1)
    ret1, frame1 = camera .read()
    j = 0

    image_1 = cv2.resize(frame, (256, 256))
    
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_1_b_w= np.dstack([image_1]*3)


    image2= cv2.resize(frame1, (256, 256))
    
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image_1_b_w1 = np.dstack([image2]* 3) 
    
    absdiff = cv2.absdiff(image_1_b_w, image_1_b_w1)
    
 
    
    
    ar = np.expand_dims(absdiff, axis=0)
    val = model2.predict(ar)
    if val == 1:
        ar = cv2.putText(ar, 'Unsigned', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        FRAME_WINDOW1.image(ar)
    else:
        ar = cv2.putText(ar, 'Signed', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        FRAME_WINDOW1.image(ar)
