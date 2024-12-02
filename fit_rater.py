from ultralytics import YOLO
import cv2
import json
import fit_metrics as metrics
import os
import numpy as np
from scipy.spatial import KDTree
import webcolors
import matplotlib.pyplot as plt


# def get_colour_name(bgr_tuple):
#     min_colours = {}
#     for name in webcolors.names("css3"):
#         r_c, g_c, b_c = webcolors.name_to_rgb(name)
#         rd = (r_c - bgr_tuple[2]) ** 2
#         gd = (g_c - bgr_tuple[1]) ** 2
#         bd = (b_c - bgr_tuple[0]) ** 2
#         min_colours[(rd + gd + bd)] = name
#     return min_colours[min(min_colours.keys())]

def get_colour_name(bgr_tuple):
    # add colors: https://www.w3schools.com/tags/ref_colornames.asp
    myColors = {
        "red"     : "#ff0000", # R
        "orange"  : "#ffa500", # O
        "yellow"  : "#ffff00", # Y
        "green"   : "#008000", # G
        "blue"    : "#0000ff", # B
        "magenta" : "#ff00ff", # I
        "purple"  : "#800080", # V
        "black"   : "#000000", # B
        "white"   : "#FFFFFF", # W
        "pink"    : "#FFC0CB",
        "grey"    : "#808080" 
        # "light blue"    : "#ADD8E6", 
        
        # # "coral"   : "#ff7f50", # light red
        # "maroon"  : "#800000", # dark red
        # "navy"    : "#000080", # dark blue
        # "cyan"    : "#00ffff", # light blue
        # "gold"    : "#ffd700", # dark yellow
        # "lime"    : "#00ff00", # bright green
        # "jade"    : "#00a36c", # light green
        # "olive"   : "#808000", # dark green
        }
        
    min_colours = {}
    for name, hex_val in myColors.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_val)
        rd = (r_c - bgr_tuple[2]) ** 2
        gd = (g_c - bgr_tuple[1]) ** 2
        bd = (b_c - bgr_tuple[0]) ** 2
        min_colours[(rd + gd + bd)] = name
    
    return min_colours[min(min_colours.keys())]

def generateRating(img, outfit):
    main_colors = []
    only_colors = []
    complexities = []
    aesthetics = {"neutral":0, "gloomy":0, "vibrant":0}
    for category, bbox in outfit:
        cropped_article = metrics.cropToBbox(img, bbox)
        complexities.append(metrics.get_complexity(cropped_article))
        colors = metrics.get_colors(cropped_article)
        # print(colors)
        main_colors.append((category, colors[0]))
        only_colors.append((colors[0]))

    i = 0
    for category, color in main_colors: 
        print(str(category) + ": " + str(get_colour_name(color)) + ". Complexity = " + str(complexities[i]) + "\n")
        i += 1
    
    print("Aesthetic: " + str(metrics.getAesthetic(only_colors)) + "\n")
    
    for color in only_colors:
        print("Gloomy: " + str(metrics.isGloomy(color)))
        print("Neutral: " + str(metrics.isNeutral(color)))
        # print("Bright: " + str(metrics.isBright(color)))
        print("Vibrant: " + str(metrics.isVibrant(color)) + "\n")
        # print("Grey: " + str(not metrics.is_not_grey(color)) + "\n")


    if len(main_colors) < 1:
        print("Nothing discernible. Dress up and try again.")
        exit()
   
    print("Overall Complexity: " + str(np.mean(complexities)*100) + " %")

    errors = []
    for _, c1 in main_colors:
        for _, c2 in main_colors:
            if not metrics.areCompatible(c1, c2): 
                errors.append((get_colour_name(c1), get_colour_name(c2)))
    print("When it comes to color theory, you made " + str(len(errors)) + " mistakes.")
    if len(errors) > 0:
        print("Those were the following color pairs: " + str(errors) + "... Maybe take some notes for next time. ")
    else:
        print("Congrats! ")

    return 

    
def rate_my_fit(filepath):
    img = cv2.imread(filepath)
    shirt_img = cv2.imread('370/lshirt1.png', cv2.IMREAD_UNCHANGED)
    pant_img = cv2.imread('370/pant8.png', cv2.IMREAD_UNCHANGED) # 1,1  3,2  1,8  
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
        
    generateRating(img, outfit)

    img2 = img

    for class_name, bbox in outfit:
        img2 = metrics.visualize_bbox(img2, bbox, class_name)
    cv2.imwrite("box_img.jpg", img2)

    for class_name, bbox in outfit:
        if class_name == "long-sleeve shirt":
            # Convert bbox to pixel coordinates
            center_x = int(bbox[0] * w)
            center_y = int(bbox[1] * h)
            box_width = int(bbox[2] * w)
            box_height = int(bbox[3] * h)
            
            # Calculate top-left corner of the bounding box
            top_left_x = int(center_x - box_width / 2)
            top_left_y = int(center_y - box_height / 2)
            
            # Resize clothing image to match bounding box dimensions
            resized_clothing = cv2.resize(shirt_img, (box_width, box_height))
            
            # Overlay the clothing image onto the person image
            # img[top_left_y:top_left_y + box_height, top_left_x:top_left_x + box_width] = resized_clothing
            # Check if the clothing image has an alpha channel
            if resized_clothing.shape[2] == 4:
                # Separate BGR and alpha channels
                clothing_bgr = resized_clothing[:, :, :3]
                clothing_alpha = resized_clothing[:, :, 3] / 255.0  # Normalize alpha to [0, 1]

                # Extract the region of interest (ROI) from the target image where clothing will be overlaid
                roi = img[top_left_y:top_left_y + box_height, top_left_x:top_left_x + box_width]

                # Blend the clothing image with the ROI in the original image using the alpha mask
                for c in range(3):  # For each color channel
                    roi[:, :, c] = clothing_bgr[:, :, c] * clothing_alpha + roi[:, :, c] * (1 - clothing_alpha)

                # Update the original image with the blended ROI
                img[top_left_y:top_left_y + box_height, top_left_x:top_left_x + box_width] = roi

            else:
                # Directly overlay if there is no transparency
                img[top_left_y:top_left_y + box_height, top_left_x:top_left_x + box_width] = resized_clothing

        if class_name == "pair of pants":
            # Convert bbox to pixel coordinates
            center_x = int(bbox[0] * w)
            center_y = int(bbox[1] * h)
            box_width = int(bbox[2] * w)
            box_height = int(bbox[3] * h)
            
            # Calculate top-left corner of the bounding box
            top_left_x = int(center_x - box_width / 2)
            top_left_y = int(center_y - box_height / 2)
            
            # Resize clothing image to match bounding box dimensions
            resized_clothing = cv2.resize(pant_img, (box_width, box_height))
            
            # Overlay the clothing image onto the person image
            # img[top_left_y:top_left_y + box_height, top_left_x:top_left_x + box_width] = resized_clothing
            # Check if the clothing image has an alpha channel
            if resized_clothing.shape[2] == 4:
                # Separate BGR and alpha channels
                clothing_bgr = resized_clothing[:, :, :3]
                clothing_alpha = resized_clothing[:, :, 3] / 255.0  # Normalize alpha to [0, 1]

                # Extract the region of interest (ROI) from the target image where clothing will be overlaid
                roi = img[top_left_y:top_left_y + box_height, top_left_x:top_left_x + box_width]

                # Blend the clothing image with the ROI in the original image using the alpha mask
                for c in range(3):  # For each color channel
                    roi[:, :, c] = clothing_bgr[:, :, c] * clothing_alpha + roi[:, :, c] * (1 - clothing_alpha)

                # Update the original image with the blended ROI
                img[top_left_y:top_left_y + box_height, top_left_x:top_left_x + box_width] = roi

            else:
                # Directly overlay if there is no transparency
                img[top_left_y:top_left_y + box_height, top_left_x:top_left_x + box_width] = resized_clothing

    return img
    # os.remove(filepath)

pwd = os.path.realpath(os.path.dirname(__file__))
names = ["short-sleeve shirt", "long-sleeve shirt", "short-sleeveoutwear", "long-sleeveoutwear", "pair of shorts", "pair of pants", "skirt", "hat", "shoe"]

model = YOLO(pwd + "/best.pt")  # load a model
have_a_model = True

cv2.imwrite("fit_img.jpg", rate_my_fit(pwd + "/bl.png"))

# DeepFashion
# https://www.kaggle.com/datasets/vishalbsadanand/deepfashion-1?resource=download
