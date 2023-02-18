import cv2
import numpy as np
import dlib
import time
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from pupil_detection_tools import *
import socket
import sys
from struct import unpack

#raspIP = ''#192.168.0.187' # IP address of (Raspberry Pi) GSR server if available
raspIP = ''
save_data = False  # Save collected data in csv file?
save_image = False # Capture and save sample images?

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

kernel = np.ones((5,5),np.uint8)

# =============== Parameters for live plot
plt.style.use('ggplot')
size = 20
t_axis = np.linspace(0,10,size+1)[0:-1]
p_size = np.zeros(len(t_axis))
r_skin = np.zeros(len(t_axis))
lumin = np.zeros(len(t_axis))

line1=[]
line2=[]
line3=[]

date_str = datetime.now().strftime('%Y-%m-%d')
todays_path = './data/' + date_str
while True:
        #time.sleep(0.5)
    _, frame = cap.read()
    #frame = cv2.resize(frame, (960,540))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
    face_frame = frame[y:y1,x:x1]
    landmarks = predictor(gray, face)
    left_point = (landmarks.part(36).x, landmarks.part(36).y)
    right_point = (landmarks.part(39).x, landmarks.part(39).y)
    center_top = midpoint(landmarks.part(37), landmarks.part(38))
    center_bottom = midpoint(landmarks.part(41), landmarks.part(40))
   
    left_eye, aperture_left = get_eye(frame, landmarks, is_left=True)
    right_eye, aperture_right = get_eye(frame, landmarks, is_left=False)
    #print(aperture)
    if aperture_left<0.2:
        print(datetime.now().strftime('%H:%M:%S.%f')[:-4]+' BLINK')
        cv2.imshow("Frame", face_frame)
        continue
    
    ratio_pupil = []
    eyes = []
    ratio_left = measure_pupil(left_eye)
    ratio_right = measure_pupil(right_eye)

    if ratio_left is not None:
        ratio_pupil.append(ratio_left)
        eyes.append(left_eye)

    if ratio_right is not None:
        ratio_pupil.append(ratio_right)
        eyes.append(right_eye)
    
    # Add values to the lists
    if len(ratio_pupil) > 0:
        ratio_pupil = min(ratio_pupil)
        time_str = datetime.now().strftime('%H:%M:%S.%f')[:-4]
        # Optional GSR server
        if any(raspIP):
            # Wait for message
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((raspIP, 5005))
            message, address = sock.recvfrom(12)#4096)
            ch0, ch1, ch2 = unpack('3f', message)
            ch0, ch1, ch2 = round(ch0,3), round(ch1,3), round(ch2,3)
            row = [time_str, ratio_pupil, ch1, ch2]
            print(time_str + f' pupil_S: {ratio_pupil}, test_R: {ch0}, photo_R: {ch1}, skin_R: {ch2}')
        else:
            print(time_str + f' Pupil size: {ratio_pupil}')
            row = [time_str, ratio_pupil]
        # ===========   Save results 
        # Save data in csv file
        if save_data:
            write_row('./data/' + date_str + '.csv',row)
        
        # Save image of processed eyes
        if save_image:
            for eye in eyes:
                write_image(todays_path,eye)
        
        if any(raspIP):
            p_size, lumin, line1, line3 = update_plot2(t_axis, p_size, lumin,line1, line3,ratio_pupil,ch1, 'Saturation', ['Pupil size','Luminance'])
            r_skin, line2 = update_plot(t_axis, r_skin, line2, ch2, 'Skin conductance (nS)')
            plt.ylim([0.3,0.5])
        else:
            p_size, line1 = update_plot(t_axis, p_size, line1, ratio_pupil, 'Pupil size (rel. to iris)')
           
   

    cv2.imshow("Frame", face_frame)
   
    #key = cv2.waitKey(1) & 0xFF == ord('q')
    if cv2.waitKey(1) & 0xFF == ord('q'): #== 27:
        break

cap.release()
cv2.destroyAllWindows()