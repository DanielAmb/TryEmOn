# from django.shortcuts import render

# def index(request):
#     return render(request, 'index.html')


# from rest_framework import generics
# from .models import User
# from .serializers import UserSerializer

# # List all users or create a new one
# class UserListCreate(generics.ListCreateAPIView):
#     queryset = User.objects.all()
#     serializer_class = UserSerializer

# # Retrieve, update, or delete a user by ID
# class UserDetail(generics.RetrieveUpdateDestroyAPIView):
#     queryset = User.objects.all()
#     serializer_class = UserSerializer

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

file_path = os.path.join('..', 'media', 'best.pt')
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
            
            boxed_img = rate_my_fit(img)

            success, buffer = cv2.imencode('.jpg', boxed_img)
            if not success:
                return Response({"error": "Image processing failed"}, status=400)
            new_image_file = ContentFile(buffer.tobytes(), image_file.name)
            
            serializer.save(image=new_image_file)


            # serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)
    

def rate_my_fit(img):
    if have_a_model:
        pred = model(img)
        outfit = []
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
    else:
        print('Error: No Model Detected.')
        exit()
        
    # text = generateRating(img, outfit)
    
    for class_name, bbox in outfit:
        img = visualize_bbox(img, bbox, class_name)
    # os.remove(filepath)
    # return img, text
    return img

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

    
from django.shortcuts import render

def index(request):
    return render(request, 'index.html')