'''WORKING PRINCIPAL
   images are load in form of array and array contaion pixels value
   (0-255) and color image contains tree channel (R,G,B)
   here firstly we convert colour image to gray scale image
   because it is easier to do calculation using single channel



   in the image we find out big differences in pixel values and then
   deceide the lanes'''




import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    slope, intercept = line_parameters
    print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1 , y1 , x2 , y2])


def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else :
            right_fit.append((slope, intercept))

        #print(left_fit)
        #print(right_fit)

    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit,axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])
    #print(left_fit_average, 'left')
    #print(right_fit_average, 'right')




def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # it used to convert color image to yar scale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # help to smoothning image helps to remove noise
    canny = cv2.Canny(blur, 50, 150)  # cv2.Canny(image,low_thresold,high_thresold) this read the difference between lower and higher thresold
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1, x2, y2 in lines:
            cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200,height), (1100, height), (550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

#image =  cv2.imread("lane.jpg")   #load image
#lane_image = np.copy(image)             #it make copy of image




cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)

    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=10, maxLineGap=1)
    average_lines = average_slope_intercept(frame, lines)

    line_image = display_lines(frame, average_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow('resu', combo_image)  # show image
    #cv2.imshow('res', line_image)  # show image
    #cv2.imshow('result', image)  # show image
    #cv2.imshow('result', cropped_image)  # show image
    #cv2.imshow('resul', canny_image)  # show image

    #plt.imshow(canny_image)
    #plt.show()
    if cv2.waitKey(1) & 0xFF== ord('q'):  # it display image infinitely until we give any input from keyboard
        break

cap.release()
cv2.destroyAllWindows()