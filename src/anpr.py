from src.engines.plate_detector import Detector
from src.engines.plate_classifier import Classifier
from src.engines.plate_recognition import OCR
from src.helpers import constants as const 
import os
import pandas as pd
import sys
import time 
from src.helpers.utility import construct_response , load_image_files , clean_up , get_directory_images_count
from src.helpers.imports import log



class anpr:
    def __init__(self):
        log.debug("anpr constructor")
        self.__detector = Detector()
        self.__classifier = Classifier()
        self.__ocr = OCR()        
        self.__plates_info = []
        
 
    def process_batch(self,is_release,images):        
        log.info('started processing batch')
        plates_list = None
        results = None

        #self.do_validation()            
        start = time.time()                        
        log.info('going to start plates extraction')
        plates_list , plates_count =  self.__detector.extract_plates(images) 
        if plates_count == 0:
            result = construct_response(plates_list,is_release)
            return result


        # classifiy images and keep result in memory for ocr processing 
        log.info('going to start classfification')         
        self.__classifier.plates_list = plates_list
        plates_classified = self.__classifier.classify_plates() 

        if plates_classified == 0:
            log.warn("no plates were classified")
            result = construct_response(self.__classifier.plates_list,is_release)
            return result



        # extract the plates numbers 
        log.info('going to start plates recognition')   
        self.__ocr.plates_list = self.__classifier.plates_list         
        self.__ocr.extract_plate_numbers_and_code()       
        end = time.time()
     
        results =  construct_response(self.__ocr.plates_list,is_release)    
        time_taken = round(end - start,2)
        log.info('done plate processing in time %d seconds',time_taken)        
        return results
        
          

    def do_validation(self):
        files =  load_image_files(const.SOURCE_FOLDER)
        assert len(files) >  0 , "no images were sent"       

    


if __name__ == '__main__': 
    plates_processor  = anpr()
    result = plates_processor.process_batch()

