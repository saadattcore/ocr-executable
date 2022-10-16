from src.helpers.imports import *
import pandas as pd
import datetime
import src.helpers.constants as const
import src.helpers.utility as util
from src.core.models import build_classification_model
from src.core.data_generators import classifier_data_generator
from src.core.losses import dice_coeff , dice_loss




class Classifier:  


    __BATCH_SIZE = 32
    __KP = 0.5
    __VAL_SPLIT = 0.2
    __TEST_SPLIT = 0.10
    __EPOCHS = 60
    __SCALE_FACTOR = 1
    __CHANNELS = 3
    __CATEGORIES = 31
    #TRAINING_PATH = DATA_PATH+IMAGES_FOLDER
 
    
    def __init__(self):   

        self.__model = None
        self.__categoryclassifier = None
        self.__load_model()
        self.__load_ksa_om_classifier()

       # self.__load_size = (412 ,104)
        self.__load_size = (256 ,64)
        self.__labeled_ground_truth = None       
        self.__images_folder = os.path.join(const.DATA_PATH, const.IMAGES_FOLDER)
        self.__plates = []

    @property
    def plates_list(self):
        return self.__plates

    @plates_list.setter
    def plates_list(self,value):
        self.__plates = value



   # ------------------ private helpers methods -----------------------------------    
    
    def __get_image_category(self,filename):
        label_series =   self.__labeled_groundtruth["Label"][csv.Filename.isin([filename])] 
        category = label_series.values[0]    
        return category
    
    def __get_image_category_vector(self,category,classes): 
        
        label = []
        for i in range(0,classes):
            label.append(0)       
            label[category] = 1   
        return label  

    def __process_predictions(self):
        country = 'United Arab Emirates'
        city = 'Dubai'
        category = ''   
        derived_categories = []         
    
        for plate in self.__plates:    
            plate_status = plate["Status"]
            if plate_status == 'Missed':
                continue
                
            country = 'United Arab Emirates'    
            prediction = plate["PredictedCategory"]       
            prediction = int(prediction)

            if prediction == 26 or prediction == 24:
                continue
			
			            
            if (prediction == 0) or (prediction == 29) or (prediction == 30):  
                city = 'Dubai'
                category = 'Private'
                
            elif  prediction == 1:  city = 'Dubai';  category='Taxi';
                    
            elif  prediction == 2:  city = 'Dubai';  category='Public Transportation';      
                    
            elif  prediction == 3:  city = 'Dubai';  category='Consulate Authority'; 
                
            elif  prediction == 4:  city = 'Dubai'; category='Police';          
                
            elif  prediction == 5:  city='Abu Dhabi';  category='Private'; 
                
            elif  prediction == 6:  city='Abu Dhabi';  category='AD 1';           
                    
            elif  prediction == 7:  city='Abu Dhabi';  category='AD 2'; 
                
            elif  prediction == 8:  city='Abu Dhabi';  category='P. Auh'; 
            
            elif  prediction == 9:  city='Abu Dhabi';  category='Taxi (Yellow)'; 
            
            elif  (prediction == 10 or prediction == 16 or prediction == 17): city='Sharjah'; category='Private'; 
                        
            elif  prediction == 11: city='Sharjah';  category='Taxi'; 
                
            elif  prediction == 12: city='Ajman';  category='Private'; 
                    
            elif  prediction == 13: city='Al Fujairah';  category='Private'; 
                
            elif  prediction == 14: city='Ras Al Khaymah';  category= 'Private';          
                    
            elif  prediction == 15: city='Um Al Quewain';   category='Private'; 
            
            elif  prediction == 18: city='Abu Dhabi';   category='Police';

            elif  prediction == 19: city='Dubai';   category='Motorcycle';

            elif  prediction == 20: city='';   category=''; country = 'Bahrain';

            elif  prediction == 21: city='';   category=''; country = 'Iraq';

            elif  prediction == 22: city='';   category=''; country = 'Kuwait';

            elif  prediction == 23: city='';   category=''; country = 'Lebanon';

            #elif  prediction == 24: city='';   category=''; country = 'Oman';

            elif  prediction == 25: city='';   category=''; country = 'Qatar';

            #elif  prediction == 26: city='';   category=''; country = 'Saudi Arabia';

            elif  prediction == 27: city='';   category=''; country = 'Syria';

            elif  prediction == 28: city='';   category=''; country = 'Yemen'; 
            
            elif prediction == 38: city='Dubai'; category='Motorcycle 2'

            else:
                city = '' ; category = '' ; country = ''





            #elif  prediction == 20: city=0;   category=0; country = 214;

            #elif  prediction == 21: city=0;   category=0; country = 208;

            #elif  prediction == 22: city=0;   category=0; country = 211;

            #elif  prediction == 23: city=0;   category=0; country = 207;

            #elif  prediction == 24: city=0;   category=0; country = 215;

            #elif  prediction == 25: city=0;   category=0; country = 213;

            #elif  prediction == 26: city=0;   category=0; country = 210;

            #elif  prediction == 27: city=0;   category=0; country = 206;

            #elif  prediction == 28: city=0;   category=0; country = 216;               
           
            
            plate["Country"] = country
            plate["Emirate"] = city
            plate["Category"] = category                    
        
        

    def __load_model(self):
        model_file =os.path.join(const.MODEL_PATH,const.CLASSIFICATION_MODEL)       
        if self.__model == None:
            log.info('loading classification model %s',model_file)
            self.__model = load_model(model_file)


    def __load_ksa_om_classifier(self):
        model_file =os.path.join(const.MODEL_PATH,const.KSA_OM_CATEGORY_CLASSIFIER)       
        if self.__categoryclassifier == None:
            log.info('loading ksa category classifier model %s',model_file)
            self.__categoryclassifier = load_model(model_file)
        
 
    #--- classification -----------------  
    def classify_plates(self):
        predictions = []      
        plates_classified = 0

        plates_found_str = [(plate["TransactionID"],plate["Plate"]) for plate in self.__plates if plate["Status"] != "Plate Not Found"]
        log.info('Total plates found %d for classification',len(plates_found_str))        

        image_str_list = [plate_image_str[1] for plate_image_str in plates_found_str]
        
        in_memory_images = util.base64_to_images(image_str_list,self.__SCALE_FACTOR,self.__CHANNELS , self.__load_size)
        #in_memory_images = util.load_images_in_memory_noreshape_gray(filepaths,self.__SCALE_FACTOR,self.__CHANNELS , self.__load_size)     
        predictions_probilities = self.__model.predict(in_memory_images)
        #cat = self.__model.predict_classes(in_memory_images)

        for index in range(0,len(predictions_probilities)):
            prob_array = predictions_probilities[index]
            category = np.argmax(prob_array)
            max_prob = math.floor(prob_array[category] * 100) 
            #print("Prob = {0} and Cat = {1}".format(max_prob,category))
            if max_prob >= 90:
                item = plates_found_str[index]
                transactionid = item[0]
                plate = ([plate for plate in self.__plates if plate["TransactionID"] == transactionid])[0]
                category = int(category)

                if category == 26 or category == 24:
                    ksa_oman_plate_category = self.__get_ksa_oman_category(plate,category)
                    if len(ksa_oman_plate_category) > 0:
                        plate["Category"] = ksa_oman_plate_category
                        plate["City"] = ""
                        plate["Country"] = "Saudi Arabia" if category == 26 else "Oman"
                    else:
                        plate["Status"] = "Missed"

                plate["PredictedCategory"] = category    
                plates_classified += 1
            else:
                item = plates_found_str[index]
                transactionid = item[0]
                plate = ([plate for plate in self.__plates if plate["TransactionID"] == transactionid])[0] 
                plate["Status"] = "Missed"
               
        print("category = {} and max probability = {}".format(int(category),max_prob))
        self.__process_predictions()
        return plates_classified


    def __get_ksa_oman_category(self,plate,platecategory):
        IMG_H = int(self.__load_size[1]/self.__SCALE_FACTOR)
        IMG_W = int(self.__load_size[0]/self.__SCALE_FACTOR)
        plates_array = np.zeros([1, IMG_H, IMG_W, self.__CHANNELS], dtype = np.uint8)
        coordinates = plate["PlateCoordinates"]
        fullimage = plate["FullImage"]        
        plateregion = None
        if platecategory == 26:
            plateregion = fullimage[coordinates[1]:coordinates[1] + (coordinates[3] + 5) , coordinates[0]:coordinates[0] + (coordinates[2] + 20)]
        else:
            plateregion = util.base64_to_image(plate["Plate"],(256,64))

        tmp = util.image_to_base64(plateregion)

        plateregion = cv2.resize(plateregion, (IMG_W,IMG_H), interpolation=cv2.INTER_AREA)
        #plateregion = util.base64_to_image(plate["Plate"],(IMG_W,IMG_H)) 
        plateregion = util.BGR2RGB(plateregion)
        plateregion = np.reshape(util.equalize_RGB(plateregion), (IMG_H, IMG_W, self.__CHANNELS))
        plates_array[0] =  plateregion        
        
        prob_array = self.__categoryclassifier.predict(plates_array)   
        prob_array = prob_array[0] 
        category = int(np.argmax(prob_array))
        probability = int(math.floor(prob_array[category] * 100))
        returnvalue = ""

        if probability > 90:
            if category == 0 or category == 2:
                returnvalue =  "Private"
            elif category == 1 or category == 3:
                returnvalue =  "Delegate"               

        return returnvalue
    




 