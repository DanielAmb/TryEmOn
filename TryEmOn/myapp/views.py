from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import ImageSerializer
from django.core.files.base import ContentFile
import numpy as np
import cv2
from io import BytesIO
from ultralytics import YOLO
import os
from cv2 import imshow, waitKey, rectangle, putText, getTextSize, FONT_HERSHEY_SIMPLEX, LINE_AA, Canny
import copy
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import colorspacious as cs
import requests, json
from sklearn.cluster import KMeans
import math

from scipy.spatial import KDTree
import webcolors
import matplotlib.pyplot as plt

import torch
from torchvision import transforms, models
from PIL import Image


file_path = os.path.join('.', 'media', 'best.pt')
file_path2 = os.path.join('.', 'media', 'fit_classifier128.pt')
names = ["short-sleeve shirt", "long-sleeve shirt", "short-sleeveoutwear", "long-sleeveoutwear", "pair of shorts", "pair of pants", "skirt", "hat", "shoe"]
model = YOLO(file_path) 
have_a_model = True
BOX_COLOR = (0, 255, 0) # Green
TEXT_COLOR = (255, 255, 255) # White

class ImageUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():

            image_file = request.FILES['image']
            image_array = np.fromstring(image_file.read(), np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            boxed_img, class_names, color_names, complexity, aesthetic, aesthetics, errors, rating, confidence = rate_my_fit(img)

            success, buffer = cv2.imencode('.jpg', boxed_img)
            if not success:
                return Response({"error": "Image processing failed"}, status=400)
            new_image_file = ContentFile(buffer.tobytes(), image_file.name)
            
            serializer.save(image=new_image_file)

            response_data = serializer.data
            response_data['class_names'] = class_names
            response_data['color_names'] = color_names
            response_data['OverallComplexity'] = complexity
            response_data['OverallAesthetic'] = aesthetic
            response_data['Aesthetics'] = aesthetics
            response_data['ColorTheoryErrors'] = errors
            response_data['ai_rating'] = rating
            response_data['confidence'] = confidence

            return Response(response_data, status=201)
        return Response(serializer.errors, status=400)
    

def rate_my_fit(img):
    outfit = []
    class_names = []

    rating, confidence = ai_rater(img)

    if have_a_model:
        pred = model(img)
        for results in pred:
            box = results.boxes.cpu().numpy()
            for b in box:
                bbox = list(b.xywh[0])
                h, w, channels = img.shape
                bbox[0] *= 1/w
                bbox[1] *= 1/h
                bbox[2] *= 1/w
                bbox[3] *= 1/h
                class_name = names[int(list(b.cls)[0])]
                if float(list(b.conf)[0]) > 0.65:
                    outfit.append((class_name, bbox))
                    class_names.append(class_name)
    else:
        print('Error: No Model Detected.')
        exit()
        
    # if(outfit == None):
    #     return None

    color_names, complexity, aesthetic, aesthetics, errors = generateRating(img, outfit) #######################################
    
    for class_name, bbox in outfit:
        img = visualize_bbox(img, bbox, class_name)
    # os.remove(filepath)
    return img, class_names, color_names, complexity, aesthetic, aesthetics, errors, rating, confidence



def generateRating(img, outfit):
    main_colors = []
    only_colors = []
    complexities = []

    for category, bbox in outfit:
        cropped_article = cropToBbox(img, bbox)
        complexities.append(get_complexity(cropped_article))
        colors = get_colors(cropped_article)

        if colors != None:
            main_colors.append((category, colors[0]))
            only_colors.append((colors[0]))

    color_names = []
    for category, color in main_colors: 
        # print(str(category) + ": " + str(get_colour_name(color)) + ": " + str(color) + "\n")
        color_names.append(str(get_colour_name(color)))

    aesthetics = []

    for color in only_colors:
        color_attributes = [
            color,
            isGloomy(color),
            isNeutral(color),
            isVibrant(color)
        ]
        aesthetics.append(color_attributes)

    errors = []
    for _, c1 in main_colors:
        for _, c2 in main_colors:
            if not areCompatible(c1, c2): 
                errors.append((get_colour_name(c1), get_colour_name(c2)))

    return color_names, (str(round(np.mean(complexities)*100, 2)) + " %"), str(getAesthetic(only_colors)), aesthetics, len(errors)

def ai_rater(img):
    # Transformation pipeline for the input image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Set up the device and load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(file_path2, weights_only=True))
    model.to(device)
    model.eval()

    import torch.nn.functional as F

    def predict_rating(img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Convert NumPy array to PIL Image
        image = Image.fromarray(image)

        # Apply the transformation pipeline
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        rating = "Bad" if predicted.item() == 0 else "Good"
        confidence_percentage = confidence.item() * 90

        return rating, confidence_percentage

    # Test the function
    # predicted_rating, confidence = predict_rating(img)
    # print(f"Predicted Fit: {predicted_rating} with Confidence: {confidence:.2f}%")
    return predict_rating(img)


# BGR (opposite of RGB, opencv image format) to color name
def get_colour_name(bgr_tuple):

    rgb_tuple = (bgr_tuple[2], bgr_tuple[1], bgr_tuple[0])
    print(rgb_tuple)
    print("get_colour_name")
    
    min_colours = {}
    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - rgb_tuple[0]) ** 2
        gd = (g_c - rgb_tuple[1]) ** 2
        bd = (b_c - rgb_tuple[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

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

#return True if basically the same color
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

#return True if colors go well together based on color theory
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

# Finds at most 5 of the most frequent colors in the bboxed image (not always accurate)
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

def get_complexity(img):
    edges = Canny(img,50,150,apertureSize = 3)
    w, h = edges.shape
    return 7*np.sum(edges)/(w*h*255)

    
from django.shortcuts import render

def index(request):
    return render(request, 'index.html')