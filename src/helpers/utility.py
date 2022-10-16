import random
import time
import os
import cv2
import csv
import ast
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import pandas as pd
from tqdm import tqdm
from scipy.ndimage.measurements import label
import json
from src.helpers import constants as const 
import base64
import re

def load_image_files(path):
    image_paths = glob.glob(path+'*.jpg')
    return image_paths


def load_annotations(file):    
    bounding_boxes = pd.read_csv(file, converters={"Box": ast.literal_eval})
    bounding_boxes.drop('Index', axis = 1, inplace=True)
    return bounding_boxes

def load_raw_image(file):
    #JPEG images are read as BGR and [0,255]
    img = cv2.imread(file)
    return img
    

def load_image(file, size):
    #JPEG images are read as BGR and [0,255]    
    img = cv2.imread(file)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

def BGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def GRAY2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def BGR2GRAY(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def draw_boxes(img, bbox, color=(0, 255, 0), thick=3):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Draw a rectangle given bbox coordinates
    cv2.rectangle(imcopy, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def resize_image(img, size):
    new_img = img.copy()
    return cv2.resize(new_img, size, interpolation=cv2.INTER_AREA)

def get_scaled_bbox(norm_bbox, size):
    scaled_bbox = (int(norm_bbox[0]*size[0]), 
                   int(norm_bbox[1]*size[1]),
                   int(norm_bbox[2]*size[0]), 
                   int(norm_bbox[3]*size[1]))
    return scaled_bbox

def get_labels(bbox_df):
    labels = []
    for index, row in bbox_df.iterrows():
        labels.append([bbox_df['x'].iloc[index], bbox_df['y'].iloc[index], bbox_df['w'].iloc[index], bbox_df['h'].iloc[index]])
    return np.asarray(labels)

def equalize_RGB(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    y, u, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YUV))
    y = clahe.apply(y)
    img = cv2.merge((y,u,v))
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    return img

def equalize(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    return clahe.apply(img)

def blur(img):
    return cv2.bilateralFilter(img,9,75,75)

def adjust_contrast(input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def process_image(file, load_size, scale_factor):
    img = load_image(file, load_size)    
    size = (int(img.shape[1]/scale_factor), int(img.shape[0]/scale_factor))
    img = resize_image(img, size)    
    #img = BGR2GRAY(img)
    #img = np.reshape(equalize(img), (img.shape[0], img.shape[1], 1))
    return img

def get_mask(img, box):
    img_mask = np.zeros_like(img[:,:,0])
    img_mask[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]= 1.
    img_mask = np.reshape(img_mask,(np.shape(img_mask)[0],np.shape(img_mask)[1],1))
    return img_mask

def get_masked_img(img, mask):
    return cv2.bitwise_and(img, img, mask = mask)

def get_image_category_vector(category,classes): 
    
    label = []
    for i in range(0,classes):
        label.append(0)  
    label[category] = 1   
    return label

def process_predictions(predictions, min_height, min_width):
    LABEL_DIM = 4
    image_width = predictions.shape[2]
    image_height = predictions.shape[1]
    #Initialize bounding boxes to zero to return zero box if no mask is found
    bounding_boxes = np.zeros([len(predictions), LABEL_DIM], dtype = np.float64)
    max_area = 0 #Initalize area to zero for first comparison
    for index, prediction in enumerate(predictions): #prediction is masked image
        prediction = np.asarray(255*prediction, dtype = np.uint8)
        pred_boxes, n_boxes = label(prediction) 
        #Labeling of boxes starts from 1
        for box_number in range(1, n_boxes+1):
            #Isolate the pixels corresponding to the box number
            #Returns two arrays containing x and y coordinates of pixels
            box_nonzero = (pred_boxes == box_number).nonzero()
            # Identify x and y values of those pixels
            box_nonzeroy = np.array(box_nonzero[0])
            box_nonzerox = np.array(box_nonzero[1])
            x = box_nonzerox.min()
            y = box_nonzeroy.min()
            w = box_nonzerox.max() - x
            h = box_nonzeroy.max() - y
            #Check if height is zero
            if (h >= min_height*prediction.shape[0]):
                #Calculate aspect ratio and area
                AR = float(w)/h
                #Set coordinates of bounding box only if AR requirements are met and it is the largest box
                if ((1.3 < AR < 6)  & 
                    (w >= min_width*prediction.shape[1]) & 
                    (x != 0) & (y != 0)):
                    bounding_boxes[index] = [float(x)/image_width, float(y)/image_height,
                                             float(w)/image_width, float(h)/image_height]            
    return bounding_boxes  


def process_prediction(prediction, min_height, min_width):
    prediction = np.asarray(255*prediction, dtype = np.uint8)
    pred_boxes, n_boxes = label(prediction)
    box = [0.0, 0.0, 0.0, 0.0]
    print("Number of boxes found:", n_boxes)
    print("WxH thresholds:",(min_width*prediction.shape[1], min_height*prediction.shape[0] ))
    for box_number in range(1, n_boxes+1):
            #Isolate the pixels corresponding to the box number
            #Returns two arrays containing x and y coordinates of pixels
            box_nonzero = (pred_boxes == box_number).nonzero()
            # Identify x and y values of those pixels
            box_nonzeroy = np.array(box_nonzero[0])
            box_nonzerox = np.array(box_nonzero[1])
            x = box_nonzerox.min()
            y = box_nonzeroy.min()
            w = box_nonzerox.max() - x
            h = box_nonzeroy.max() - y
            print("Box values:", x,y,w,h)
            #Check if height is zero
            if (h >= min_height*prediction.shape[0]):
                #Calculate aspect ratio and area
                AR = float(w)/h
                #Set coordinates of bounding box only if AR requirements are met and it is the largest box
                if ((1.3 < AR < 6)  & 
                    (w >= min_width*prediction.shape[1]) & 
                    (x != 0) & (y != 0)):
                    box = [x,y,w,h]
    return box


def extract_plate(img, box):
        return img[box[1]:box[1]+box[3], box[0]:box[0]+box[2], :]


def load_images_in_memory(filepaths,scale_factor, img_depth,load_size):    
        n_val_images = len(filepaths)
        IMG_H = int(load_size[1]/scale_factor)
        IMG_W = int(load_size[0]/scale_factor)
        IMG_D = img_depth
        LABEL_DIM = 4
        val_images = np.zeros([n_val_images, IMG_H, IMG_W, IMG_D], dtype = np.uint8)

        for index, file in tqdm(enumerate(filepaths), total = len(filepaths)):
            image = process_image(file,load_size, scale_factor)
            image = BGR2GRAY(image)
            image = np.reshape(equalize(image), (IMG_H, IMG_W, img_depth))
            val_images[index] = image

        return val_images

def detector_base64_to_images(filepaths,scale_factor, img_depth,load_size):    
        n_val_images = len(filepaths)
        IMG_H = int(load_size[1]/scale_factor)
        IMG_W = int(load_size[0]/scale_factor)
        IMG_D = img_depth
        LABEL_DIM = 4
        val_images = np.zeros([n_val_images, IMG_H, IMG_W, IMG_D], dtype = np.uint8)

        for index, file in tqdm(enumerate(filepaths), total = len(filepaths)):
            img = base64_to_image(file,load_size)            
            size = (int(img.shape[1]/scale_factor), int(img.shape[0]/scale_factor))
            img = resize_image(img, size)            
            img = BGR2GRAY(img)
            image = np.reshape(equalize(img), (IMG_H, IMG_W, img_depth))
            val_images[index] = image

        return val_images

def load_images_in_memory_noreshape(filepaths,scale_factor, img_depth,load_size):    
        n_val_images = len(filepaths)
        IMG_H = int(load_size[1]/scale_factor)
        IMG_W = int(load_size[0]/scale_factor)
        IMG_D = img_depth
        LABEL_DIM = 4
        val_images = np.zeros([n_val_images, IMG_H, IMG_W, IMG_D], dtype = np.uint8)

        for index, file in tqdm(enumerate(filepaths), total = len(filepaths)):            
            img = cv2.imread(file)           
            img = cv2.resize(img, load_size, interpolation=cv2.INTER_AREA)              
            size = (int(img.shape[1]/scale_factor), int(img.shape[0]/scale_factor))             
            new_img = img.copy()
            image = cv2.resize(new_img, size, interpolation=cv2.INTER_AREA)            
            image = equalize_RGB(image)
            image = BGR2RGB(image)           
            val_images[index] = image

        return val_images

def load_images_in_memory_noreshape_gray(filepaths,scale_factor, img_depth,load_size):    
        n_val_images = len(filepaths)
        IMG_H = int(load_size[1]/scale_factor)
        IMG_W = int(load_size[0]/scale_factor)
        IMG_D = img_depth
        LABEL_DIM = 4
        val_images = np.zeros([n_val_images, IMG_H, IMG_W, IMG_D], dtype = np.uint8)

        for index, file in tqdm(enumerate(filepaths), total = len(filepaths)):            
            img = cv2.imread(file)           
            img = cv2.resize(img, load_size, interpolation=cv2.INTER_AREA)              
            size = (int(img.shape[1]/scale_factor), int(img.shape[0]/scale_factor))             
            new_img = img.copy()
            image = cv2.resize(new_img, size, interpolation=cv2.INTER_AREA)  
            image = BGR2GRAY(image)   
            image = np.reshape(equalize(image), (IMG_H, IMG_W, img_depth))  
            val_images[index] = image

        return val_images


def base64_to_images(base64plates,scale_factor, img_depth,load_size):
    n_val_images = len(base64plates)
    IMG_H = int(load_size[1]/scale_factor)
    IMG_W = int(load_size[0]/scale_factor)
    IMG_D = img_depth   
    val_images = np.zeros([n_val_images, IMG_H, IMG_W, IMG_D], dtype = np.uint8)

    for index in range(len(base64plates)):  
        img = base64_to_image(base64plates[index],load_size)  
        size = (int(img.shape[1]/scale_factor), int(img.shape[0]/scale_factor))        
        image = cv2.resize(img, size, interpolation=cv2.INTER_AREA)  
        if IMG_D == 1:
            image = BGR2GRAY(image)  
            image = np.reshape(equalize(image), (IMG_H, IMG_W, img_depth))  
        elif IMG_D == 3:
            image = BGR2RGB(image)
            image = np.reshape(equalize_RGB(image), (IMG_H, IMG_W, img_depth))  
        
        val_images[index] = image
    return val_images
    
def image_to_base64(plate_image):
    is_successfull , encoded_image = cv2.imencode(".jpg",plate_image)  
    assert is_successfull == True , "Image encoding failed"                
    encoded_bytes  = base64.b64encode(encoded_image)  
    image_str = encoded_bytes.decode("utf-8")
    return image_str

def base64_to_image(image_str,size):
    image_bytes = base64.b64decode(image_str)
    image_array = np.fromstring(image_bytes,dtype=np.uint8)        
    img = cv2.imdecode(image_array, cv2.IMREAD_ANYCOLOR)  
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def construct_response(ocr_results,is_release):
    json_data = []

    for plate in ocr_results:
        del plate["FullImage"] 
           
        #tarnsactionid = plate["TransactionID"]
        #if is_release:
         #   tmp_tr_id = tarnsactionid[0:tarnsactionid.find(".")]            
          #  tran_id = int(tmp_tr_id)
        #else:
         #   filename = tarnsactionid
    

        platenumber = plate['PlateNumber']
        color = ''
        predicted_category = plate["PredictedCategory"]
        status = plate["Status"]
        category_desc = ''

        if status == "Recognized":                  
       
            # 1 -  DUBAI TAXI
            # 10 - SHJ SL
            # 11 - SHJ TAXI
            # 16 - SHJ ML
            # 17 - SHJ WHITE

            lookup_numeric_plates = [1,10,11,16,17]

            if predicted_category in lookup_numeric_plates:
                if not platenumber.isnumeric() or len(platenumber) > 6:
                    plate["Status"] = 'Missed'
                    plate["PlateNumber"] = ''
                    plate["Color"] = ''
                    plate["ConfidenceLevel"] = 0
                    plate["Country"] = ''
                    plate["Emirate"] = ''
                    plate["Category"] = ''
                    json_data.append(plate)
                    continue  

            if predicted_category == 26:
                ismatched = re.match("^\d+[a-zA-Z]{3}$",platenumber)                    
                if ismatched == None:
                    plate["Status"] = 'Missed'
                    plate["PlateNumber"] = ''
                    plate["Color"] = ''
                    plate["ConfidenceLevel"] = 0
                    plate["Country"] = ''
                    plate["Emirate"] = ''
                    plate["Category"] = ''
                    json_data.append(plate)
                    continue          


            # 0  - DUBAI PRIVATE
            # 10 - SHJ SL
            # 30 - DUBAI SL 
            # 12 - AJMAN PRIVATE
            # 13 - FUJAIRAH PRIVATE
            # 14 - RAK PRIVATE 
            # 15 - UMALQUEIN PRIVATE
            # 29 - DUBAI ML
            # 16 - SHJ ML

            private_category_lookup = [0,10,30,12,13,14,15,29]
            exclude_color_lookup = [0,30,12,13,14,15,29]
            cat_exclude_list = [5]
            
            if predicted_category in (private_category_lookup):                                    
                if predicted_category in cat_exclude_list:
                    continue



                color = platenumber[0]
                platenumber = platenumber[1:len(platenumber)]
                if((predicted_category in exclude_color_lookup and (color.isdigit() or len(color) == 0)) or (predicted_category in [10,16] and (not color in ['1','2','3']))):
                    plate["Status"] = 'Missed'
                    plate["PlateNumber"] = ''
                    plate["Color"] = ''
                    plate["ConfidenceLevel"] = 0
                    plate["Country"] = ''
                    plate["Emirate"] = ''
                    plate["Category"] = ''
                    json_data.append(plate)
                    continue
                
            
            if predicted_category in ([5]):

                plt_array = platenumber.split(' ')

                if len(plt_array) == 1:
                    plate["Status"] = 'Missed'
                    plate["PlateNumber"] = ''
                    plate["Color"] = ''
                    plate["ConfidenceLevel"] = 0
                    plate["Country"] = ''
                    plate["Emirate"] = ''
                    plate["Category"] = ''
                    json_data.append(plate)
                    continue                                       


                color = plt_array[0]
                platenumber = plt_array[1]

                if (not platenumber.isnumeric()) or (not color.isnumeric()) or (len(color) == 0):
                    plate["Status"] = 'Missed'
                    plate["PlateNumber"] = ''
                    plate["Color"] = ''
                    plate["ConfidenceLevel"] = 0
                    plate["Country"] = ''
                    plate["Emirate"] = ''
                    plate["Category"] = ''
                    json_data.append(plate)
                    continue


            if predicted_category == 17:
                color = "WH"       
            
            if predicted_category == 3:
                color = 0 

            #emirates_private_categories_lookup = [0,5,12,13,10,16,14,15]

            #if (predicted_category in emirates_private_categories_lookup) and (color == 0 or len(color) == 0):
             #   plate["Status"] = 'Missed'
             #   plate["PlateNumber"] = ''
             #   plate["Color"] = ''
             #   plate["ConfidenceLevel"] = 0
             #   plate["Country"] = ''
             #   plate["Emirate"] = ''
             #   plate["Category"] = ''
             #   json_data.append(plate)
                
        
            category_desc = __get_category_description(predicted_category)

        j_object = {}

        
        if  is_release == False:
            j_object["ImageName"] = plate["ImageName"] 
            j_object["PredictedCategory"] = predicted_category
            j_object["FullCategory"] = category_desc
   

        j_object["TransactionID"] = plate["TransactionID"]     
        j_object["PlateNumber"] = platenumber
        j_object["Color"] = color
        j_object["Country"] = plate["Country"]
        j_object["Emirate"] = plate["Emirate"]
        j_object["Category"] = plate["Category"]    
        j_object["ConfidenceLevel"] = plate["ConfidenceLevel"]
        j_object["Status"] = plate["Status"]           
        j_object["Plate"] = plate["Plate"]   

        json_data.append(j_object)

    return json_data

def __get_category_description(predicted_category):
    cat_desc = ''
    if predicted_category == 0:
        cat_desc = "Dubai Private"
    elif predicted_category == 1:
        cat_desc = "Dubai Taxi"
    elif predicted_category == 2:
        cat_desc = "Dubai Public Transportation"
    elif predicted_category == 3:
        cat_desc = "Dubai Consulate Authority"
    elif predicted_category == 4:
        cat_desc = "Dubai Police"
    elif predicted_category == 5:
        cat_desc = "Abu Dhabi Private"
    elif predicted_category == 6:
        cat_desc = "Abu Dhabi AD 1"
    elif predicted_category == 7:
        cat_desc = "Abu Dhabi AD 2"
    elif predicted_category == 8:
        cat_desc = "Abu Dhabi P.Auth"
    elif predicted_category == 9:
        cat_desc = "Abu Dhabi Taxi (Yellow)"
    elif predicted_category == 10:
        cat_desc = "Sharjah Private Single Line"
    elif predicted_category == 11:
        cat_desc = "Sharjah Taxi"
    elif predicted_category == 12:
        cat_desc = "Ajman Private"
    elif predicted_category == 13:
        cat_desc = "Al Fujairah Private"
    elif predicted_category == 14:
        cat_desc = "Ras Al Khaimah Private"
    elif predicted_category == 15:
        cat_desc = "Um Al Quwain Private"
    elif predicted_category == 16:
        cat_desc = "Sharjah Private Multiline"
    elif predicted_category == 17:
        cat_desc = "Sharjah Private White"
    elif predicted_category == 18:
        cat_desc = "Abu Dhabi P0lice"
    elif predicted_category == 19:
        cat_desc = "Motorcycle"
    elif predicted_category == 20:
        cat_desc = "Bahrain"
    elif predicted_category == 21:
        cat_desc = "Iraq"
    elif predicted_category == 22:
        cat_desc = "Kuwait"
    elif predicted_category == 23:
        cat_desc = "Libya"
    elif predicted_category == 24:
        cat_desc = "Oman"
    elif predicted_category == 25:
        cat_desc = "Qatar"
    elif predicted_category == 26:
        cat_desc = "KSA"
    elif predicted_category == 27:
        cat_desc = "Syria"
    elif predicted_category == 28:
        cat_desc = "Yeman"
    elif predicted_category == 29:
        cat_desc = "Dubai Private Multiline"
    elif predicted_category == 30:
            cat_desc = "Dubai Private Singleline"

    return cat_desc    

def clean_up():
    files = glob.glob(const.SOURCE_FOLDER + "*.jpg")
    for file in files:
        os.remove(file)

    files = glob.glob(const.PROCESSED_PLATES_FOLDER + "*.jpg")
    for file in files:
        os.remove(file)
    
    files = glob.glob(const.COLOR_CODE_FOLDER + "*.jpg")
    for file in files:
        os.remove(file)
    
    files = glob.glob(const.MULTILINE_PLATE_NUMBER_FOLDER + "*.jpg")
    for file in files:
        os.remove(file)

def get_directory_images_count(path):
    return len(glob.glob(path + "*.jpg"))

def add_miss_plates(json_data,dic_plates_bounds_list,is_release,key,tran_id):
    bounds = dic_plates_bounds_list[key]
    j_object = {}

    if is_release:
        j_object["TransactionID"] = tran_id                  
    else:
        j_object["ImageName"] = key
        j_object["PredictedCategory"] = -1
        j_object["FullCategory"] = ""

    j_object["PlateNumber"] = ""
    j_object["Color"] = ""
    j_object["Country"] = ""
    j_object["Emirate"] = ""
    j_object["Category"] = ""
    j_object["ConfidenceLevel"] = 0.0
    j_object["Status"] = "Missed"    
    json_data.append(j_object)

def crop_ksa_plate_region(base64platestring):
    plate = base64_to_image(base64platestring,(256,64))
    h , w = plate.shape[0:2]
    midpoint = int(h/2)
    midpoint = midpoint - 5
    license_plate_part = plate[midpoint:midpoint + h,0:w]
    crop_plate_str = image_to_base64(license_plate_part)
    return crop_plate_str


    