import math
from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle
from src.helpers.utility import process_image , get_scaled_bbox , get_mask,BGR2GRAY,equalize , get_image_category_vector
from keras.preprocessing.image import ImageDataGenerator


def data_generator(X_samples, y_samples, batch_size, load_size, scale_factor, channels):
    n_samples = len(X_samples)
    IMG_H = int(1024/scale_factor) #scale the load_size down
    IMG_W = int(2560/scale_factor) #scale the load_size down
    IMG_D = channels
    LABEL_DIM = 4
    n_batches = math.ceil(n_samples/batch_size)
    
    while 1: 
        for offset in tqdm(range(0, n_samples, batch_size), total = n_batches):
            #Set y_batch to extracted normalized boxes
            boxes = y_samples[offset:offset+batch_size]
            files = X_samples[offset:offset+batch_size] #Get the filenames for the batch
            #Declare zero arrays to hold the images
            X_batch = np.zeros([batch_size, IMG_H, IMG_W, IMG_D], dtype = np.uint8)
            y_batch = np.zeros([batch_size, IMG_H, IMG_W, 1], dtype = np.uint8)

            #Loop over all files to read images and append to output arrays
            for index, (box, file) in enumerate(zip(boxes, files)):
                image = process_image(file, load_size, scale_factor)
                scaled_box = get_scaled_bbox(box, (IMG_W, IMG_H))
                y_batch[index] = get_mask(image, scaled_box)
                image = BGR2GRAY(image)
                image = np.reshape(equalize(image), (IMG_H, IMG_W, channels))
                X_batch[index] = image
    
            yield shuffle(X_batch, y_batch)

def classifier_data_generator(X_samples, y_samples, batch_size, load_size, scale_factor, channels,categories):
    n_samples = len(X_samples)
    #IMG_H = int(1024/scale_factor) #scale the load_size down
    #IMG_W = int(2560/scale_factor) #scale the load_size down
    
    IMG_H = int(load_size[1]/scale_factor) #scale the load_size down
    IMG_W = int(load_size[0]/scale_factor) #scale the load_size down
    
    #IMG_H = img_height
    #IMG_W = img_width

    IMG_D = channels
    LABEL_DIM = 4
    n_batches = math.ceil(n_samples/batch_size)
    
    while 1: 
        for offset in tqdm(range(0, n_samples, batch_size), total = n_batches):
            #Set y_batch to extracted normalized boxes
            boxes = y_samples[offset:offset+batch_size]
            files = X_samples[offset:offset+batch_size] #Get the filenames for the batch
            #Declare zero arrays to hold the images
            X_batch = np.zeros([batch_size, IMG_H, IMG_W, IMG_D], dtype = np.float64)
           # y_batch = np.zeros([batch_size, IMG_H, IMG_W, 1], dtype = np.uint8)
            y_batch = np.zeros([batch_size,categories], dtype = np.uint8)

            #Loop over all files to read images and append to output arrays
            for index, (box, file) in enumerate(zip(boxes, files)): 
                #print(file)
                image = process_image(file, load_size, scale_factor)
                #scaled_box = get_scaled_bbox(box, (IMG_W, IMG_H))
               # print(box)
                y_batch[index] = get_image_category_vector(box , categories) 
              #  print(y_batch[index])
               # print(file)
                
                image = BGR2GRAY(image)
                image = np.reshape(equalize(image), (IMG_H, IMG_W, channels))
                X_batch[index] = image               
      
                
            datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=80,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True            
            )           
            
               
            datagen.fit(X_batch)                
    
            yield (X_batch, y_batch)            