from src.helpers.imports import *
import src.helpers.constants as const
import src.helpers.utility as util
from src.core.models import build_engine
from src.core.data_generators import data_generator
from src.core.losses import dice_coeff , dice_loss
from src.helpers.constants import TRAINING_DATA_PATH , GT_FILE, EXTENSION , TRAINING_PLATES_FOLDER ,VALIDATION_PLATES_FOLDER ,DATA_TATTILE,TEST_PLATES_FOLDER , TOTAL_PLATES_FOLDER , MODEL_PATH, JAI_PLATE_PATH,TATTILE_PLATE_PATH
import matplotlib.pyplot as plt
import sys


class OCR: 

    __EPOCHS = 60
    __BATCH_SIZE = 4
    __MAX_PLATE_CHARS = 6
    __LEARNING_RATE = 1e-3
    __AUG_FACTOR = 4
    __DROP_PROB = 0.25

    #alphabet = u'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ '

    def __init__(self):
        self.__singleline_ocr = None      
        self.__multiline_ocr = None
        self.__dubai_nt_ocr = None          
        self._plate_color_classifier = None
        self.__dubai_ml_color_classifier = None
        self.__plate_color_code_ocr = None
        self.__othertype_ocr = None

        self.__load_singleline_ocr()
        self.__load_multiline_model()
        #self.__load_plate_color_classifier()     
        self.__load_dubai_nt_model()   

        self.__load_othertype_ocr()
        #self.__load_dubai_ml_color_classifier()
        #self.__load_color_code_ocr()
        self.__plates = None


        #self.__load_size = (256, 64)
        #self.__load_size = (120, 40)
        self.__results = None

    def __load_singleline_ocr(self):        
            model_file = MODEL_PATH+ const.OCR_MODEL
            log.info('loading singline plate recognition model %s',model_file)
            OCR_engine = load_model(model_file, custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
            OCR_input_layer = OCR_engine.get_layer(name='Input').input
            OCR_output_layer = OCR_engine.get_layer(name='softmax').output
            self.__singleline_ocr = K.function([OCR_input_layer, K.learning_phase()], [OCR_output_layer])       
    
    def __load_multiline_model(self):
            model_file = MODEL_PATH+ const.MULTILINE_OCR
            log.info('loading multiline plate recognition model %s',model_file)
            OCR_engine = load_model(model_file, custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
            OCR_input_layer = OCR_engine.get_layer(name='Input').input
            OCR_output_layer = OCR_engine.get_layer(name='softmax').output
            self.__multiline_ocr = K.function([OCR_input_layer, K.learning_phase()], [OCR_output_layer])

    def __load_othertype_ocr(self):
        model_file = MODEL_PATH+ const.KSA_OM_BIKE_OCR_MODEL
        log.info('loading singline plate recognition model %s',model_file)
        OCR_engine = load_model(model_file, custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
        OCR_input_layer = OCR_engine.get_layer(name='Input').input
        OCR_output_layer = OCR_engine.get_layer(name='softmax').output
        self.__othertype_ocr = K.function([OCR_input_layer, K.learning_phase()], [OCR_output_layer])       

    def __load_plate_color_classifier(self):
            model_file = MODEL_PATH+ const.PLATE_COLOR_CLASSIFIER
            if self._plate_color_classifier == None:
                log.info('loading classification model %s',model_file)
                self._plate_color_classifier = load_model(model_file)

    def __load_dubai_ml_color_classifier(self):
            model_file = MODEL_PATH+ const.DUBAI_ML_PLATE_COLOR_CLASSIFIER
            if self.__dubai_ml_color_classifier == None:
                log.info('loading classification model %s',model_file)
                self.__dubai_ml_color_classifier = load_model(model_file)

    def __load_color_code_ocr(self):
            model_file = MODEL_PATH+ const.CODE_OCR
            if self.__plate_color_code_ocr == None:
                OCR_engine = load_model(model_file, custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
                OCR_input_layer = OCR_engine.get_layer(name='Input').input
                OCR_output_layer = OCR_engine.get_layer(name='softmax').output
                log.info('loading color code ocr  %s',model_file)
                self.__plate_color_code_ocr = K.function([OCR_input_layer, K.learning_phase()], [OCR_output_layer])
          
    def __load_dubai_nt_model(self):
            model_file = MODEL_PATH+ const.DUBAI_NT_OCR
            log.info('loading plate color code recognition model %s',model_file)
            OCR_engine = load_model(model_file, custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
            OCR_input_layer = OCR_engine.get_layer(name='Input').input
            OCR_output_layer = OCR_engine.get_layer(name='softmax').output
            self.__dubai_nt_ocr = K.function([OCR_input_layer, K.learning_phase()], [OCR_output_layer]) 
   

    @property
    def plates_list(self):
        return self.__plates

    @plates_list.setter
    def plates_list(self,value):
        self.__plates = value


    #------ helpers -----------------------------

    @staticmethod
    def transform_image(img, rt_range, xlate_range):
        #Define Transformations - rotation and translation
        if img.ndim == 3:
            pixels_x, pixels_y, channels = img.shape    
        else: 
            pixels_x, pixels_y = img.shape
        rt_angle = np.random.uniform(rt_range) - rt_range/2
        xlate_x = np.random.uniform(xlate_range) - xlate_range/2
        xlate_y = np.random.uniform(xlate_range) - xlate_range/2
        
        M_rot = cv2.getRotationMatrix2D((pixels_y/2, pixels_x/2), rt_angle, 1)
        M_xlate = np.float32([[1,0,xlate_x],[0,1,xlate_y]])
        
        #Transform image
        img = cv2.warpAffine(img, M_rot, (pixels_y, pixels_x))
        img = cv2.warpAffine(img, M_xlate, (pixels_y, pixels_x))
        
        return img

    @staticmethod
    def augment_image(img, aug_factor):
        ROT_RANGE = 7
        XLATE_RANGE = 10
        if img.ndim == 3:
            images = np.zeros([aug_factor, img.shape[0], img.shape[1], img.shape[2]], dtype = np.uint8)
            images[0] = img
            for i in range(aug_factor-1):
                images[i+1] =OCR.transform_image(img, ROT_RANGE, XLATE_RANGE)
        else: 
            images = np.zeros([aug_factor, img.shape[0], img.shape[1], 1], dtype = np.uint8)
            images[0] = img.reshape([img.shape[0], img.shape[1], 1])
            for i in range(aug_factor-1):
                xformed_img = OCR.transform_image(img, ROT_RANGE, XLATE_RANGE)
                images[i+1] = xformed_img.reshape(img.shape[0], img.shape[1], 1)  
        return images
    
    @staticmethod
    def process_image(file, size, aug_factor):
        img = util.load_image(file, size)
        img = util.BGR2GRAY(img)
        images = OCR.augment_image(img, aug_factor)
        return images.transpose(0,2,1,3) #Returns array of images of shape (aug_factor, img_w, img_h, img_d)
    
    @staticmethod
    def process_tattile_image(self,file, size, aug_factor):
        img = LoadImage(file, size)
        img = BGR2GRAY(img)
        #img = img[:,:,0]
        images = augment_image(img, aug_factor)
        return images.transpose(0,2,1,3) #Returns array of images of shape (aug_factor, img_w, img_h, img_d)
   
    @staticmethod
    def text_to_labels(text,alphabets):
        ret = []
        for char in text:
            ret.append(int(alphabets.find(char)))
        return ret

    @staticmethod
    def labels_to_text(labels,alphabets):
        ret = []
        for c in labels:
            if c == len(alphabets):  # CTC Blank
                ret.append("")
            else:
                ret.append(alphabets[c])
        return "".join(ret)

    @staticmethod
    def get_accuracy(results):
        return results['Match'].sum()/len(results)*100    
      
    #----- predictions---------------------
   
    
    #--- plate number extraction ---
    # plate_type 
    # 1 means old single line plate
    # 2 means dubai private new types plates

    def extract_plate_numbers_and_code(self):
                   
        single_line_plates = []
        multi_line_plates = []
        single_line_dubai_nt_plates = []  
        other_type_plates = []
        number_of_plates_recognized = 0

        # 0  - DUBAI PRIVATE
        # 1  - DUBAI TAXI
        # 10 - SHJ SL 
        # 30 - DUBAI SL !! 
        # 12 - AJMAN PRIVATE
        # 13 - FUJAIRAH PRIVATE
        # 14 - RAK PRIVATE 
        # 15 - UMALQUEIN PRIVATE
        # 29 - DUBAI ML
        # 17 - SHJ WHITE
        # 16 - SHJ ML !!
        # 11 - SHJ TAXI

        singleline__plate_source_lookup = [0,1,10,17,12,13,14,15,29,11]        
        multiline_plate_source_lookup = [5]
        other_types_source_lookup = [38,19,26,24]

        #singleline__plate_source_lookup = [0,1,10,17,29,11]        
        #multiline_plate_source_lookup = []


        for plate in self.__plates:
            status = plate["Status"]
            if status == "Missed":
                continue

            t_file = plate["TransactionID"]            
            plate_category = plate["PredictedCategory"] 
            if plate_category in multiline_plate_source_lookup:                
                multi_line_plates.append((plate["TransactionID"],plate["Plate"]))
            elif (plate_category in singleline__plate_source_lookup):                
                single_line_plates.append((plate["TransactionID"],plate["Plate"]))
            elif plate_category == 30:
                single_line_dubai_nt_plates.append((plate["TransactionID"],plate["Plate"]))
            elif (plate_category in other_types_source_lookup):
                if plate_category == 26:
                    cropped_plate = util.crop_ksa_plate_region(plate["Plate"])                     
                    other_type_plates.append((plate["TransactionID"],cropped_plate))
                else:                                    
                    other_type_plates.append((plate["TransactionID"],plate["Plate"]))
            else:
                status = plate["Status"]
                if not status:
                    plate["Status"] = "Missed"                     
            
        log.info("Number of single line plates are %d" , len(single_line_plates) )
        log.info("Number of multiline plates are %d" , len(multi_line_plates)) 

        if len(single_line_plates) > 0:
            self.__extract_singleline_plates_numbers(single_line_plates,1)
        if len(multi_line_plates) > 0:            
            self.__extract_abudhabi_plates_numbers(multi_line_plates)
        if len(single_line_dubai_nt_plates) > 0:
            self.__extract_singleline_plates_numbers(single_line_dubai_nt_plates,2)   
        if len(other_type_plates) > 0:
            self.__extract_other_plates_numbers(other_type_plates)   
            
        
        #count_list = [plate for plate in self.__plates if plate["Status"] == "Recognized"]
        #return len(count_list)

      
    
    def __extract_singleline_plates_numbers(self,single_line_plates,plate_type = 1):
        alphabet = u'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ '       
        batch_size = len(single_line_plates)   
        if plate_type == 1:
            sample_data = DataGenerator(single_line_plates,'', 'JAI', (256,64), batch_size, self.__MAX_PLATE_CHARS,self.__AUG_FACTOR,alphabet, 1)
            sample_data.read_images()
            result = sample_data.decode_set(self.__singleline_ocr,self.__plates,False) 
                   
        elif plate_type == 2:
            sample_data = DataGenerator(single_line_plates,'', 'JAI', (256,64), batch_size, self.__MAX_PLATE_CHARS,self.__AUG_FACTOR,alphabet, 3)
            sample_data.read_images()
            result = sample_data.decode_set(self.__dubai_nt_ocr,self.__plates,False)

        log.info("Exiting __extract_singleline_plates_numbers function")        
        

    def __extract_abudhabi_plates_numbers(self,multi_line_plates):
        alphabet = u'0123456789 '       
        batch_size = len(multi_line_plates) 
        
        sample_data = DataGenerator(multi_line_plates,'', 'JAI', (256,64), batch_size, 8,self.__AUG_FACTOR,alphabet, 1)
        sample_data.read_images()
        sample_data.decode_set(self.__multiline_ocr,self.__plates,True)   

        log.info("Exiting __extract_abudhabi_plates_numbers function")      

    def __extract_other_plates_numbers(self,other_plates):
        alphabet = u'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ '              
        batch_size = len(other_plates) 
        
        sample_data = DataGenerator(other_plates,'', 'JAI', (256,64), batch_size, 8,self.__AUG_FACTOR,alphabet, 1)
        sample_data.read_images()
        sample_data.decode_set(self.__othertype_ocr,self.__plates,True)   

        log.info("Exiting __extract_abudhabi_plates_numbers function")     
       

    def __extract_multiline_plates_numbers(self,multi_line_plates):  
        
        image_files = [file[0] for file in multi_line_plates]

        result = []        
        plates_colors_aud_shj = []
        plates_colors_dubai_nt = []        
        alphabet = u'0123456789 '       

        batch_size = len(multi_line_plates)
        plate_number_data_generator = DataGenerator(image_files,'', 'JAI', (256,64), batch_size, self.__MAX_PLATE_CHARS,self.__AUG_FACTOR,alphabet,False)
        plate_number_data_generator.read_images()
        plates_numbers = plate_number_data_generator.decode_set(self.__multiline_ocr,self.__plates,False)  


        for index,row in enumerate(multi_line_plates):

            image = cv2.imread(plates_folder +  row[0])
            h , w = image.shape[:2]  
            plate = None         

            if(row[4] == 5 or row[4] == 16):
                ratio = w / h  
                if ratio >= 1 and ratio <= 2.5:
                    plate = image[0:60 , 0:w]                    
                elif ratio >= 2.5:
                    plate = image[0:h , 0:60] 
            elif(row[4] == 29):
                cal_h = math.ceil(h / 2)
                cal_h = cal_h +  8    
                plate = image[0:cal_h , 0:w]

            cv2.imwrite(const.COLOR_CODE_FOLDER + os.path.basename(row[0]),plate)    

        alphabet_code = u'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ '
        batch_size = len(multi_line_plates)
        plate_code_data_generator = DataGenerator(image_files, const.COLOR_CODE_FOLDER, 'JAI', (180, 64), batch_size, 2,self.__AUG_FACTOR,alphabet_code,False,1)
        plate_code_data_generator.read_images()
        plate_colors = plate_code_data_generator.decode_set(self.__plate_color_code_ocr,False)          

        dic_plate_numbers = dict((element[0] , element[1:]) for element in plates_numbers)
        dic_color_codes = dict((element[0] , element[1:]) for element in plate_colors)

        for key in dic_plate_numbers.keys():
            if key in dic_color_codes:
                plt_num = dic_plate_numbers[key]
                plt_color = dic_color_codes[key]
                conf = min(plt_num[3] , plt_color[3])
                result.append((key , plt_num[0],plt_color[0] , plt_num[1] , plt_num[2] , conf , plt_color[1]))       
                
        return result 

  
    
    def __get_plate_color(self,multi_line_plates,plate_folder,is_dubai_ml_plates):
        color_codes = []
        predictions_probilities = None
        processed_pridictions = []
           
        file_paths = [plate_folder + file for file in multi_line_plates]
        log.info('Total plates found %d for color classification',len(file_paths))    

        if not is_dubai_ml_plates:    
            masked_plates = self.__mask_plate_number(file_paths) 
            predictions_probilities = self._plate_color_classifier.predict(masked_plates)
        else:
            in_memory_images = util.load_images_in_memory(file_paths,1,1,(360,121))
            predictions_probilities = self.__dubai_ml_color_classifier.predict(in_memory_images)

        #cat = self._plate_color_classifier.predict_classes(masked_plates)

        for index in range(0,len(predictions_probilities)):
            prob_array = predictions_probilities[index]
            color_category = np.argmax(prob_array)
            max_prob = math.floor(prob_array[color_category] * 100) 
            if max_prob >= 90:
                color_category = int(color_category)
                processing_file = multi_line_plates[index]
                category_info = [category_info for category_info in self.plate_categories if category_info[0] == processing_file]
                plate_category = category_info[0][4] 
                if not is_dubai_ml_plates:
                    if self.__validate_color_code(plate_category,color_category):
                        color_codes.append((os.path.basename(file_paths[index]),color_category)) 
                        processed_pridictions = self.__parse_predicticted_color_codes(color_codes)   
                else:
                    color_codes.append((os.path.basename(file_paths[index]),color_category))
                    processed_pridictions = self.__parse_dubai_ml_codes(color_codes)

        
        return processed_pridictions

    def __mask_plate_number(self,file_paths):
        #in_memory_images = util.load_images_in_memory(file_paths,1,1,(360,121)) 
        masked_images = np.zeros([len(file_paths),121,360,1],dtype=np.uint8)
        for index , file in enumerate(file_paths):
            image = util.load_raw_image(file)
            h , w = image.shape[:2]
            ratio = w / h
            #image = util.load_image(file,(360,121))                    

            if ratio >= 1 and ratio <= 2.5:
                 plate = image[47:h , 0:w]                    
            elif ratio >= 2.5:
                plate = image[0:h , 47:w] 
                    
            mser = cv2.MSER_create()
            regions = mser.detectRegions(plate,None)  
            for r in regions:
                 (x,y,w,h) = cv2.boundingRect(r)
                 if ratio >= 1 and ratio <= 2.5:
                     image[y+51:(y+51)+h , x:x+w] =  0
                 elif ratio >= 2.5:
                     image[y:y+h , x + 51:(x + 51)+w] =  0
            #plt.imshow(image)
            #plt.show()
           
            
            #image =  np.reshape(util.equalize(image), (121, 360, 1))
            img = cv2.resize(image, (360,121), interpolation=cv2.INTER_AREA)  
            img = util.BGR2GRAY(img) 
            img =  np.reshape(util.equalize(img), (121, 360, 1))
            cv2.imwrite(const.COLOR_CODE_FOLDER + os.path.basename(file),img)
            masked_images[index] = img
        return masked_images
    
    def __validate_color_code(self,plate_category,color_category):
        result = False
        if ((color_category >=1 and color_category <=18) and (plate_category == 5)) or ((color_category >=19 and color_category <= 21) and (plate_category == 16)) or ((color_category >= 22 and color_category <= 47) and (plate_category == 29)):
            result = True
        return result

    def __parse_dubai_ml_codes(self,predictions):
        result = []
        color_code = "" 
        file = ""      

        for prediction in predictions:
            file = prediction[0]
            predicted_color = prediction[1]

            if predicted_color == 0:
                color_code = "A"
            elif predicted_color == 1:
                color_code = "B"
            elif predicted_color == 2:
                color_code = "C"
            elif predicted_color == 3:
                color_code = "D"
            elif predicted_color == 4:
                color_code = "E"
            elif predicted_color == 5:
                color_code = "F"
            elif predicted_color == 6:
                color_code = "G"
            elif predicted_color == 7:
                color_code = "H"
            elif predicted_color == 8:
                color_code = "I"
            elif predicted_color == 9:
                color_code = "J"
            elif predicted_color == 10:
                color_code = "K"
            elif predicted_color == 11:
                color_code = "L"
            elif predicted_color == 12:
                color_code = "M"
            elif predicted_color == 13:
                color_code = "N"
            elif predicted_color == 14:
                color_code = "O"
            elif predicted_color == 15:
                color_code = "P"
            elif predicted_color == 16:
                color_code = "Q"
            elif predicted_color == 17:
                color_code = "R"
            elif predicted_color == 18:
                color_code = "S"
            elif predicted_color == 19:
                color_code = "T"
            elif predicted_color == 20:
                color_code = "U"
            elif predicted_color == 21:
                color_code = "V"
            elif predicted_color == 22:
                color_code = "W"
            elif predicted_color == 23:
                color_code = "X"                
            elif predicted_color == 24:
                color_code = "Y"                
            elif predicted_color == 25:
                color_code = "Z"                
            
            result.append((file,color_code,predicted_color))
        return result

    def __parse_predicticted_color_codes(self,predictions):
        result = []
        color_code = "" 
        file = ""      

        for prediction in predictions:
            file = prediction[0]
            predicted_color = prediction[1]

            if predicted_color == 1:
                color_code = "1"
            elif predicted_color == 2:
                color_code = "2"
            elif predicted_color == 3:
                color_code = "3"
            elif predicted_color == 4:
                color_code = "4"
            elif predicted_color == 5:
                color_code = "5"
            elif predicted_color == 6:
                color_code = "6"
            elif predicted_color == 7:
                color_code = "7"
            elif predicted_color == 8:
                color_code = "8"
            elif predicted_color == 9:
                color_code = "9"
            elif predicted_color == 10:
                color_code = "10"
            elif predicted_color == 11:
                color_code = "11"
            elif predicted_color == 12:
                color_code = "12"
            elif predicted_color == 13:
                color_code = "13"
            elif predicted_color == 14:
                color_code = "14"
            elif predicted_color == 15:
                color_code = "15"
            elif predicted_color == 16:
                color_code = "16"
            elif predicted_color == 17:
                color_code = "17"
            elif predicted_color == 18:
                color_code = "50"
            elif predicted_color == 19:
                color_code = "1"
            elif predicted_color == 20:
                color_code = "2"
            elif predicted_color == 21:
                color_code = "3"
            elif predicted_color == 22:
                color_code = "A"
            elif predicted_color == 23:
                color_code = "B"
            elif predicted_color == 24:
                color_code = "C"
            elif predicted_color == 25:
                color_code = "D"
            elif predicted_color == 26:
                color_code = "E"
            elif predicted_color == 27:
                color_code = "F"
            elif predicted_color == 28:
                color_code = "G"
            elif predicted_color == 29:
                color_code = "H"
            elif predicted_color == 30:
                color_code = "I"
            elif predicted_color == 31:
                color_code = "J"
            elif predicted_color == 32:
                color_code = "K"
            elif predicted_color == 33:
                color_code = "L"
            elif predicted_color == 34:
                color_code = "M"
            elif predicted_color == 35:
                color_code = "N"
            elif predicted_color == 36:
                color_code = "O"
            elif predicted_color == 37:
                color_code = "P"
            elif predicted_color == 38:
                color_code = "Q"
            elif predicted_color == 39:
                color_code = "R"
            elif predicted_color == 40:
                color_code = "S"
            elif predicted_color == 41:
                color_code = "T"
            elif predicted_color == 42:
                color_code = "U"
            elif predicted_color == 43:
                color_code = "V"
            elif predicted_color == 44:
                color_code = "W"
            elif predicted_color == 45:
                color_code = "X"                
            elif predicted_color == 46:
                color_code = "Y"                
            elif predicted_color == 47:
                color_code = "Z"                
            result.append((file,color_code,predicted_color))
        return result
        
    def __split_multiline_plates_number_code(self,plates_folder,multi_line_plates):
        #split multiline plate into code and plate number and save it to their respective folders
        for plate in multi_line_plates:
            image = util.load_raw_image(plates_folder + plate)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            h , w = image.shape[:2]
            ratio = w / h

            #color code region cropping and save it to disk 
            if ratio >= 1 and ratio <= 2.5:                       
                color_code = image[0:47 , 0:w]
            elif ratio >= 2.5:         
                color_code = image[0:h , 0:47]  

            borderType = cv2.BORDER_CONSTANT
            top = int(0.15 * image.shape[0])  # shape[0] = rows
            bottom = 0
            left = int(0.09 * image.shape[1])  # shape[1] = cols
            right = 0
            dst = cv2.copyMakeBorder(color_code, top, bottom, left, right, borderType, None, (0,0,0))   
            cv2.imwrite(const.COLOR_CODE_FOLDER +  os.path.basename(plate) ,cv2.cvtColor(dst,cv2.COLOR_BGR2RGB))  

            #mark code area with red color to make it invisble for OCR plate text
            gray = cv2.cvtColor(color_code,cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(gray,127,255,0)
            _,ctn,_ = cv2.findContours(thresh , cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)
            for con in ctn:                   
                x , y , w, h = cv2.boundingRect(con)
                image[y:y+h, x:x+w ] = (142, 84, 69)
            
            dst = cv2.copyMakeBorder(image, top, bottom, left, right, borderType, None, (0,0,0))            
            cv2.imwrite(const.MULTILINE_PLATE_NUMBER_FOLDER +  os.path.basename(plate) ,cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)) 
                   
    def __assemble_output(self,singleline_plates,single_lines_plates_categories , multiline_plates,multi_line_plates_category):        
        plate_numbers = []

        if len(multiline_plates) > 0 and len(multi_line_plates_category) > 0:
            dic_plate_numbers = dict((element[0] , element[1:]) for element in multiline_plates)
            dic_plate_category = dict((element[0] , element[1:]) for element in multi_line_plates_category)
            for key in dic_plate_numbers.keys():
                if key in dic_plate_category:
                    plt_num = dic_plate_numbers[key]
                    plt_category = dic_plate_category[key]
                    #plate_numbers.append((key , plt_num[0],plt_num[1] , plt_category[0] , plt_category[1] , plt_category[2] , plt_category[3] , plt_num[2] , plt_num[3] , plt_num[4],plt_num[5]))
                    plate_numbers.append((key , plt_num[0], '' , plt_category[0] , plt_category[1] , plt_category[2] , plt_category[3] , plt_num[1] , plt_num[2] , plt_num[3],''))        

        if len(singleline_plates) > 0 and len(single_lines_plates_categories) > 0:
            list_number = []
            list_cats = []
            for index , item in enumerate(singleline_plates):
                f = item[0]
                r = item[1:]
                list_number.append((f,r))

            for index , item in enumerate(single_lines_plates_categories):
                f = item[0]
                r = item[1:]
                list_cats.append((f,r))

            dic_plate_numbers  = dict(list_number)
            dic_plate_category  = dict(list_cats)

            for key in dic_plate_numbers.keys():
                if key in dic_plate_category:
                    plt_num = dic_plate_numbers[key]
                    plt_category = dic_plate_category[key]
                    plate_numbers.append((key , plt_num[0], '' , plt_category[0] , plt_category[1] , plt_category[2] , plt_category[3] , plt_num[1] , plt_num[2] , plt_num[3],''))        

        return plate_numbers



class DataGenerator:
        def __init__(self, data, path, src, image_size=(256,64), batch_size=64, max_plate_chars=6, aug_factor=4,alphabets='0123456789 ',image_channels = 3):
            
            self.img_w = image_size[0]
            self.img_h = image_size[1]
            self.img_d = image_channels       
            self.path = path
            self.src = src
            self.batch_size = batch_size
            self.alphabets = alphabets
            self.n_samples = batch_size
            
            self.total_samples = aug_factor * self.n_samples # Total # of samples after augmentation
            self.max_plate_chars = max_plate_chars # Max # of characters in a plate
            self.downsample_factor = 4 # The amount by the input image is scaled down prior to being fed into the LSTM layer
            self.aug_factor = aug_factor # Turn n_samples into aug_factor * n_samples by doing random translations/rotations 
      
            self.image_files = data
                    
            #Contains resized images read in from image_files by calling self.read_images()
            self.images = np.zeros([self.n_samples, self.img_h, self.img_w, self.img_d], dtype = np.uint8)      
                    
        def __get_probs(self, grouped_labels, probs):
            char_probs = []
            offset = 0
            for index, group in enumerate(grouped_labels):
                end = offset + len(group)
                prob = sum(probs[offset:end])/len(group)
                offset += len(group)
                if (37 not in group) & (38 not in group):
                    char_probs.append(prob)
            return np.asarray(char_probs, dtype = np.float32)

        def __get_probs_space(self, grouped_labels, probs):
            char_probs = []
            offset = 0
            for index, group in enumerate(grouped_labels):
                end = offset + len(group)
                prob = sum(probs[offset:end])/len(group)
                offset += len(group)
                if (11 not in group):
                    char_probs.append(prob)
            return np.asarray(char_probs, dtype = np.float32)
                
        #Read in all images 
        def read_images(self):
            if self.src == 'JAI':
                for index, file in tqdm(enumerate(self.image_files), total = self.n_samples):
                    #self.images[index] = LoadImage(self.path+file, (self.img_w, self.img_h))
                    #img = util.load_image(self.path+file, (self.img_w, self.img_h))					
                    img = util.base64_to_image(file[1],(256,64))                    
                    if(self.img_d == 3):
                        img = util.BGR2RGB(img)
                    elif(self.img_d == 1):
                        img = util.BGR2GRAY(img)                        
                        #img = util.equalize(img)
                        #img = util.blur(img)
                        img = img.reshape([img.shape[0], img.shape[1], 1])
                    #img = img.reshape([img.shape[0], img.shape[1], 3])
                    self.images[index] = img                         

            elif self.src == 'Tattile':
                for index, file in tqdm(enumerate(self.image_files), total = self.n_samples):
                    img = util.load_image(self.path+file, (self.img_w, self.img_h))
                    img = img[:,:,0]
                    self.images[index] = util.load_image(self.path+file, (self.img_w, self.img_h))
            
        #Prepare next batch of data for training/validation
        def next_batch(self):
            n_batches = self.total_samples / self.batch_size
            step_size = int(self.batch_size / self.aug_factor)
            while True:  
                #Step through images such that step * self.aug_factor = self.batch_size
                for offset in range(0, self.n_samples, step_size):
                    #Setup the elements of the input data
                    images = np.zeros([self.batch_size, self.img_w, self.img_h, self.img_d], dtype = np.uint8)
                    labels = np.ones([self.batch_size, self.max_plate_chars]) * -1 #Setup labels to be -1 (space character)
                    label_length = np.zeros([self.batch_size, 1])
                    input_length = np.ones([self.batch_size, 1]) * (self.img_w // self.downsample_factor - 2) #-2 from decode_batch
                    #Extract images * aug_factor = batch_size
                    files = self.image_files[offset:offset+step_size]
                    results = self.OCR_results[offset:offset+step_size]
                    #Populate the input data arrays
                    for index, (file, result) in enumerate(zip(files, results)):
                        start_index = index * self.aug_factor
                        stop_index = index * self.aug_factor + self.aug_factor
                        #Set the images in steps of self.aug_factor
                        if self.src == 'JAI':
                            images[start_index:stop_index] = OCR.process_image(self.path+file, 
                                                                        (self.img_w, self.img_h), self.aug_factor)
                        elif self.src == 'Tattile':
                            images[start_index:stop_index] = OCR.process_tattile_image(self.path+file, 
                                                                        (self.img_w, self.img_h), self.aug_factor)
                        #Set the first n characters of the label for each self.aug_factor images
                        labels[start_index:stop_index, 0:len(result)] = OCR.text_to_labels(result) 
                        #Set the length of the label for each self.aug_factor images
                        label_length[start_index:stop_index] = len(result)

                    #Put it all in a dict. Keys of dict must correspond to layer names in model
                    inputs = {'Input': images,
                            'Labels': labels,
                            'Input_length': input_length,
                            'Label_length': label_length}
                    outputs = {'ctc': np.zeros([self.batch_size])}
                    yield (inputs, outputs)
        
        #Get a random batch of data
        def get_batch(self):
            batch_offset = np.random.randint(len(self.image_files) - self.batch_size)
            files = self.image_files[batch_offset:batch_offset+self.batch_size]
            results = self.OCR_results[batch_offset:batch_offset+self.batch_size]
            images = np.zeros([self.batch_size, self.img_w, self.img_h, self.img_d], dtype = np.uint8)
            labels = np.ones([self.batch_size, self.max_plate_chars], dtype = np.int32) * -1
            for index, (file, result) in enumerate(zip(files, results)):
                images[index] = util.load_image(self.path+file, (self.img_w, self.img_h)).transpose(1,0,2)
                labels[index, 0:len(result)] = OCR.text_to_labels(result)
            return images, labels


        def split_abudhabi_plate_number(self,timestep_maxprob_labels):
            for idx in range(0,len(timestep_maxprob_labels)):
                               
                if(timestep_maxprob_labels[idx] == 37):
                    space_counts = space_counts +  1
                else:
                    start_index = idx
                    space_counts = 0
                
                if(space_counts >= 20):                                  
                    break
                    
            
            print(start_index)
            ndix = start_index + 1
            print(ndix)
            timestep_maxprob_labels[ndix] = 36
            return timestep_maxprob_labels



        def decode_set(self,predict,plates,plate_with_space = False):
            IGNORE_CHARS = 2          
            result_list = []            
            batch = 64
            img_count = len(self.image_files)            
            iterations = math.ceil(img_count / batch)
            for i in range(0 , iterations):
                s_index = i * batch
                e_index = s_index + batch
                #print(i)
                r_images = self.images[s_index:e_index]
                r_files = self.image_files[s_index:e_index]
                batch_results = predict([r_images.transpose(0,2,1,3), 0])[0] 
                #print("Shape of batch_results is:", batch_results.shape)
                for index, sample in enumerate(batch_results):
                    #Get the index of the probability at each timestep which corresponds to the highest prob label
                    timestep_maxprob_labels = np.argmax(sample[IGNORE_CHARS:], 1)

                    #Get the probabilities at these index values (using fancy indexing)
                    timestep_maxprob_probs = sample[np.arange(IGNORE_CHARS, len(sample)), timestep_maxprob_labels] #sample(rows,cols)
                    #Eliminate repetitions
                    maxprob_labels = [k for k, g in itertools.groupby(timestep_maxprob_labels)]
                    #print("Maxprob labels are:", maxprob_labels)
                    #Convert labels to text               
                    prediction = OCR.labels_to_text(maxprob_labels,self.alphabets)
                    if prediction == '':
                        continue
                    #print("Prediction is", prediction)
                    #Group the highest prob labels according to the label value
                    grouped_labels = [list(g) for k, g in itertools.groupby(timestep_maxprob_labels)]
                    #print("Grouped labels are:", grouped_labels)
                    #Get the probability for each character
                    if(plate_with_space):
                        label_probs = self.__get_probs_space(grouped_labels, timestep_maxprob_probs)
                    else:                                                           
                        label_probs = self.__get_probs(grouped_labels, timestep_maxprob_probs)

                    mean_prob = label_probs.mean()           
                    min_prob = min(label_probs)

                    plate_founded = [plate for plate in plates if (plate["TransactionID"] == r_files[index][0])]

                    assert len(plate_founded) == 1 , "Plate not found in decode set function"

                    plate_founded = plate_founded[0]
                    plate_founded["PlateNumber"] = prediction
                    plate_founded["ConfidenceLevel"] = float(min_prob)
                    plate_founded["Status"] = "Recognized"
                   


                    #result_list.append((r_files[index],prediction,float(mean_prob), label_probs.tolist(),float(min_prob)))                     
                #print('for iteration  size of result list' ,i,len(result_list))                     
            
                    
            #print('result set lenght %d',len(result_list))
            #return result_list
    
        


