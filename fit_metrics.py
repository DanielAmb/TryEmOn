from cv2 import imshow, waitKey, rectangle, putText, getTextSize, FONT_HERSHEY_SIMPLEX, LINE_AA, Canny
import os
import copy
import numpy as np
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import colorspacious as cs
import requests, json
from sklearn.cluster import KMeans
import math
import cv2

BOX_COLOR = (0, 255, 0) # Green
TEXT_COLOR = (255, 255, 255) # White

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    img = copy.deepcopy(img)
    x_center, y_center, w, h = bbox
    height, width, colors = img.shape
    w *= width
    h *= height
    x_center *= width
    y_center *= height
    x_min = x_center - w/2
    y_min = y_center - h/2
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
    rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = getTextSize(class_name, FONT_HERSHEY_SIMPLEX, 0.5, 1)    
    rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=FONT_HERSHEY_SIMPLEX,
        fontScale=0.5, 
        color=TEXT_COLOR, 
        lineType=LINE_AA,
    )
    return img

def cropToBbox(img, bbox):
    x_center, y_center, w, h = bbox
    height, width, colors = img.shape
    w *= width
    h *= height
    x_center *= width
    y_center *= height
    x_min = x_center - w/2
    y_min = y_center - h/2
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    crop_img = img[y_min:y_max, x_min:x_max]
    return crop_img

def is_not_grey(color):
    hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)
    return hsv_color[0][0][1] > 50  # Filter out colors with low saturation

def get_colors(img):

    flat_img = np.reshape(img, (-1, 3))

    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(flat_img)

    cluster_centers = kmeans.cluster_centers_

    percentages = (np.unique(kmeans.labels_,return_counts=True)[1])/flat_img.shape[0]

    p_and_c = list(zip(percentages,cluster_centers))

    sortedPC = sorted(p_and_c, key=lambda x: x[0], reverse=True)

    colors = []
    for p,c in sortedPC:
        if p > 0.2:
            colors.append(c)

    return colors

def is_not_grey(color):
    hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)
    return hsv_color[0][0][1] > 50  # Filter out colors with low saturation

def get_complexity(img):
    edges = Canny(img,50,150,apertureSize = 3)
    #imshow("cropped", edges)
    #waitKey(0)
    w, h = edges.shape
    return 7*np.sum(edges)/(w*h*255)

#return True if basically the same color
#otherwise return False
def areTheSame(c1, c2, tolerance=20):    
    # Convert RGB to XYZ color space
    def rgb_to_xyz(rgb):
        b, g, r = [x / 255.0 for x in rgb]
        r = r ** 2.2
        g = g ** 2.2
        b = b ** 2.2
        
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        return (x, y, z)

    # Calculate Delta E
    def delta_e(c1, c2):
        x1, y1, z1 = rgb_to_xyz(c1)
        x2, y2, z2 = rgb_to_xyz(c2)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    delta_e_value = delta_e(c1, c2)
    
    return (delta_e_value * 100) < tolerance

# Sum of the min & max of (a, b, c)
def hilo(a, b, c):
    if c < b: b, c = c, b
    if b < a: a, b = b, a
    if c < b: b, c = c, b
    return a + c

def complement(b, g, r):
    k = hilo(b, g, r)
    return tuple(k - u for u in (b, g, r))

#return True if colors go together (will say that colors that don't go together are any two of RGB)
#otherwise return False
def areCompatible(c1, c2):
    if areTheSame((255, 0, 0), c1, 40):
        if areTheSame((255, 0, 0), c2, 40): return True
        elif areTheSame((0, 255, 0), c2, 40): return False
        elif areTheSame((0, 0, 255), c2, 40): return False
    elif areTheSame((0, 255, 0), c1, 40):
        if areTheSame((255, 0, 0), c2, 40): return False
        elif areTheSame((0, 255, 0), c2, 40): return True
        elif areTheSame((0, 0, 255), c2, 40): return False
    elif areTheSame((0, 0, 255), c1, 40):
        if areTheSame((255, 0, 0), c2, 40): return False
        elif areTheSame((0, 255, 0), c2, 40): return False
        elif areTheSame((0, 0, 255), c2, 40): return True
    return True

def bgr_to_hsv(c):
    b, g, r = c
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

def isGloomy(color, tolerance=40):
    h,s,v = bgr_to_hsv(color)
    return (v < tolerance)

def isNeutral(color, tolerance=65):
    h,s,v = bgr_to_hsv(color)
    return (s < tolerance)

def isVibrant(color):
    return isBright(color) and (not isNeutral(color))

def isBright(color, tolerance=60):
    h,s,v = bgr_to_hsv(color)
    return (v >= tolerance)

def getAesthetic(colors, tolerance=0.5):
    aesthetics = [0, 0] #vibrant or gloomy
    for c in colors:
        if isVibrant(c): aesthetics[0] += 1
        elif isGloomy(c): aesthetics[1] += 1
    aesthetics[0] *= 1/len(colors)
    aesthetics[1] *= 1/len(colors)
    if aesthetics[0] > tolerance: return "vibrant"
    elif aesthetics[1] > tolerance: return "gloomy"
    else: return "neutral"