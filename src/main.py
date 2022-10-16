import os
import core
import cv2
from helpers import constants as const
from engines.plate_classifier import Classifier   
from anpr import Anpr




def test_platedetector():   

    images_path = os.path.join('data','training-plates','additional_images\\') 
    model_file = c.MODEL_PATH + 'plate_detector_Unet_v4.11--0.944.h5'   
    plates_save_location = os.path.join(c.DATA_PATH , 'plates\\ocrx plates\\')
    model_file = os.path.join('model' , 'plate_detector_Unet_v4.11--0.944.h5')
    classifier = plate_detector.Detector()
    classifier.do_model_training()
    print('finish extraction')


def test_plateclassifier():
    plate_classifier = Classifier()
    images_folder_to_classify = os.path.join(const.DATA_PATH , 'additional_plates\\')
    model_file_to_load = os.path.join(const.MODEL_PATH , 'cnn_plate_classifier_v1.79-0.250.h5')
    plate_classifier.do_image_classification(model_file_to_load,images_folder_to_classify)

    print('Done doing classification for images')
    


if __name__ == '__main__':    
    plates_processor  = Anpr()
    plates_processor.process_batch()
    #plate_processor.do_ocr_training()
    #plates_processor.extract_platenumber()
    #plates_processor.train_platedetector()
    #plates_processor.extract_plates()
    #plates_processor.classify_plates()

    
    
        


    

    
     
  


