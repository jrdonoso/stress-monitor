import cv2
import numpy as np
import dlib
import time
from datetime import datetime
from matplotlib import pyplot as plt
import os
import csv

font = cv2.FONT_HERSHEY_SIMPLEX

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def hypot(x,y):
    return np.sqrt(x**2 + y**2)

def distance(p1,p2):
    return(np.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2))

#def get_aperture(eye_point, landmarks):

def get_blink_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y) #changed
    right_point = (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    aperture = ver_line_lenght / hor_line_lenght #changed
    return aperture 

def find_circles(input_img, normalize = False, canny_thres = 50):
        img = input_img.copy()
        #img = cv2.bitwise_not(img)
        #kernel = np.ones((3,3),np.uint8)
        #img = cv2.GaussianBlur(img,(5,5),10)
        #img = cv2.erode(img,kernel,iterations = 2)
        norm_img = img#[:,:,0] # better for pupil detection
        if normalize:
                normalizedImg = np.zeros(img.shape)
                norm_img = cv2.normalize(norm_img,normalizedImg,0,255, cv2.NORM_MINMAX) #better for iris detection?
        try:
            circles = cv2.HoughCircles(norm_img, cv2.HOUGH_GRADIENT, 
                    1, #inverse ratio of resolution? (default 1)
                    32.0, # 32.0min distance between detected centers(default 10000) use 50 for iris
                    param1 = canny_thres, #upper threshold for Canny edge detector (50) use 100 for iris
                    param2 = 30, # accum threshold for center detection (30).The smaller it is, the more false circles may be detected
            #minRadius = 0, maxRadius = 0,
                    )
        except Exception as e:
            print('/!\ Circle detection error!')
            return None
        return circles


def get_pupil(img):
    normalizedImg = np.zeros(img.shape)
    norm_img = cv2.normalize(img,normalizedImg,0,255, cv2.NORM_MINMAX)
    #x_mean = norm_img.mean(0)
    #y_mean = norm_img.mean(1)
    x_min = norm_img.min(0)
    y_min = norm_img.min(1)
    # Localise dark zones and estimate radius and center
    dark_thres = 5
    x_idx = np.array(range(len(x_min)))
    y_idx = np.array(range(len(y_min)))
    x_dark = x_idx[x_min<dark_thres]
    y_dark = y_idx[y_min<dark_thres]
    radius = (len(x_dark) + len(y_dark))/4
    center = [x_dark.mean(),y_dark.mean()]
    #img = cv2.circle(img,(img.shape[1]/2,img.shape[0]/2),radius,(0,0,255),1)
    #circ_pupil = np.array([[[x_ir, y_ir, r_pupil]]])
    return radius, center

def write_image(out, frame):
    """
    writes frame from the webcam as png file to disk. datetime is used as filename.
    """
    if not os.path.exists(out):
        os.makedirs(out)
    now = datetime.now() 
    dt_string = now.strftime("%H-%M-%S_%f")[:-4]
    filename = f'{out}/{dt_string}.png'
    #logging.info(f'write image {filename}')
    cv2.imwrite(filename, frame)

def write_row(fname,row):
    with open(fname, 'a') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerow(row)

def draw_circles(circles,img, color = (0,255,0)):
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                if(i[2])>0: #and i[2]<55):
                    cv2.circle(img,(i[0],i[1]),i[2],color,1) #(B,G,R)

def is_pupil(circle, img):
    r_iris = np.round(img.shape[0]/2)
    #xi, yi, = np.round(img.shape[1]/2), np.round(img.shape[0]/2)
    xp, yp, rp = circle[0,0,0],circle[0,0,1],circle[0,0,2] 
    print(f"center_difference: /\X = {np.abs(r_iris-xp)} ; /\Y = {np.abs(r_iris-yp)}" )
    print(f"Relative radius: {rp/r_iris}")

def get_eye(frame, landmarks, is_left = False):
    eyeframe_ext = 10
    lm_set = [36, 39, 38, 40]
    if is_left:
        lm_set = [42, 45, 43, 47]
    ex1, ex2 = landmarks.part(lm_set[0]).x, landmarks.part(lm_set[1]).x
    ey1, ey2 = landmarks.part(lm_set[2]).y - eyeframe_ext, landmarks.part(lm_set[3]).y + eyeframe_ext
    # Measure apperture
    aperture = distance(landmarks.part(lm_set[2]),landmarks.part(lm_set[3])) / distance(landmarks.part(lm_set[0]),landmarks.part(lm_set[1]))
    cv2.rectangle(frame,(ex1,ey1),(ex2,ey2),(0,255,0),2)
    if aperture<0.2:
        cv2.putText(frame,"BLINK",(ex1,ey1), font, 0.3,(0,0,255))
    return frame[ey1:ey2,ex1:ex2], aperture

def get_iris(roi_color):
    roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    circ_iris = find_circles(roi_gray, normalize=True, canny_thres=100)
    iris = None
    if circ_iris is not None: 
        #print("Iris detected")
        x_ir, y_ir, r_ir   = int(circ_iris[0,0,0]), int(circ_iris[0,0,1]), int(circ_iris[0,0,2])
        # Get iris subimage
        iris = roi_gray[y_ir - r_ir : y_ir + r_ir + 1, x_ir - r_ir : x_ir + r_ir + 1]
        #print(iris.shape)
        #cv2.bitwise_not(iris)
        if iris.shape[0]==0:
            iris = None
    return iris, circ_iris

def get_pupil(iris):
    """ 
    Find pupil in iris subimage and mesures it w/r to iris 
    """
    pupir_ratio, pupir_distance, pupir_d, = None, None, None
    circ_pupil = find_circles(iris, normalize=True, canny_thres=100)
            #r_pupil, __ = get_pupil(iris)
            #circ_pupil = np.array([[[x_ir, y_ir, r_pupil]]])
    
    if circ_pupil is not None: 
        #print("-----> Pupil detected")
        #is_pupil(circ_pupil,iris)
        ###  Measure pupil w/r to iris
        #x_ir, y_ir, r_ir   = int(circ_iris[0,0,0]), int(circ_iris[0,0,1]), int(circ_iris[0,0,2])
        
        xp, yp, rp = circ_pupil[0,0,0], circ_pupil[0,0,1], circ_pupil[0,0,2] 
        r_iris = np.ceil(iris.shape[0]/2) # {rp/r_iris}
        pupir_ratio = np.round(rp/r_iris,2)
        pupir_distance = np.round(np.sqrt((r_iris-xp)**2 + (r_iris-yp)**2),2)
        pupir_d = np.round(pupir_distance/r_iris,2)
    return circ_pupil, pupir_ratio, pupir_distance, pupir_d
 

def measure_pupil(roi_color):

    iris, circ_iris = get_iris(roi_color)
    pupir_ratio = 1
    circ_pupil = None
    if iris is None:
        return None
    else:
        ###  Measure pupil w/r to iris
        circ_pupil, pupir_ratio, pupir_distance, pupir_d = get_pupil(iris)
        
    if circ_pupil is None:
        return None 
    else:
        ###  
        
        xp, yp, rp = circ_pupil[0,0,0],circ_pupil[0,0,1],circ_pupil[0,0,2] 
       
        r_iris = np.ceil(iris.shape[0]/2)
        if (pupir_ratio<0.7) and ((pupir_distance + rp) < r_iris + 4 ) and (pupir_distance < rp) and (pupir_d<0.2):
            # draw pupil and add text
            x_ir, y_ir, r_ir   = int(circ_iris[0,0,0]), int(circ_iris[0,0,1]), int(circ_iris[0,0,2])
            iris_color = roi_color[y_ir - r_ir : y_ir + r_ir + 1, x_ir - r_ir : x_ir + r_ir +1]
            draw_circles(circ_iris,roi_color)
            draw_circles(circ_pupil, iris_color, color = (0,0,255))
            cv2.putText(roi_color,f"R:{pupir_ratio}",(1,10), font, 0.3,(0,0,255))#,bottomLeftOrigin=True)#,2,cv2.LINE_AA)
            cv2.putText(roi_color,f"d:{pupir_d}",(1,50), font, 0.3,(0,0,255))
            cv2.imshow('Pupils',roi_color)
            return pupir_ratio

def live_plotter(x_vec,y1_data,line1,identifier='', n_lines=1):#,y_label = 'Pupil size (rel. to iris)'):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(6,3))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec, y1_data ,'-o', alpha=0.8)
        if any(identifier):
            ax.legend([identifier])        
        #update plot label/title
        #plt.ylabel(y_label)
        plt.xlabel('Last 10 samples')
        plt.title('Stress Monitor')# {}'.format(identifier))
        plt.show()
    
    # after the figure, axis, and line are created, update the y-data
    line1.set_ydata(y1_data)
    #plt.ylim([0,1.0])
    # adjust limits if new data goes beyond bounds
    #if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
    #    plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    #plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1

def live_plotter2(x_vec,y1_data,y2_data,line1,line2,identifiers=''):#,y_label = 'Pupil size (rel. to iris)'):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(6,3))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec, y1_data ,'-o', alpha=0.8)
        line2, = ax.plot(x_vec, y2_data ,'-o', alpha=0.8)
        if any(identifiers):
            ax.legend(identifiers)        
        #update plot label/title
        #plt.ylabel(y_label)
        plt.xlabel('Last 10 samples')
        plt.title('Stress Monitor')# {}'.format(identifier))
        plt.show()
    
    # after the figure, axis, and line are created, update the y-data
    line1.set_ydata(y1_data)
    line2.set_ydata(y2_data)
    plt.ylim([0,1.0])
    # adjust limits if new data goes beyond bounds
    #if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
    #    plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    #plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1, line2

def update_plot(t_axis, y_data, line_, new_val, y_label):
    y_data[-1] = new_val
    line_ = live_plotter(t_axis,y_data,line_)#,y_label='Skin conductance (nS)')
    plt.ylabel(y_label)
    y_data = np.append(y_data[1:],0.0)
    return y_data, line_

def update_plot2(t_axis, y_data, y_data2, line_, line_2, new_val, new_val2, y_label,identifiers):
    y_data[-1] = new_val
    y_data2[-1] = new_val2
    line_, line_2 = live_plotter2(t_axis,y_data, y_data2,line_, line_2,identifiers)#,y_label='Skin conductance (nS)')
    plt.ylabel(y_label)
    y_data = np.append(y_data[1:],0.0)
    y_data2 = np.append(y_data2[1:],0.0)
    return y_data, y_data2, line_, line_2
