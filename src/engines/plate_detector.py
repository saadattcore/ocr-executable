from src.helpers.imports import *
import src.helpers.constants as const
import src.helpers.utility as util
from src.core.models import build_small_UNet
from src.core.data_generators import data_generator
from src.core.losses import dice_coeff , dice_loss
from src.core.accuracy_metrics import get_IOU , get_mean_IOU
import base64


class Detector:  

    __BATCH_SIZE = 16
    __KP = 0.5
    __VAL_SPLIT = 0.15
    __TEST_SPLIT = 0.10
    __EPOCHS = 20
    __SCALE_FACTOR = 4
    __CHANNELS = 1   

    def __init__(self):    
        self.__load_size = (2560, 1024)
        self.__model = None
        self.__load_model()
        self.__min_width = None
        self.__min_height = None
        self.__data_dictionary = None
        self.__training_history = None
        self.__setup_data()

    #-------- private helpers methods  ------------------  

    def __load_model(self):
        model_file = model_file = os.path.join('model' , const.DETECTION_MODEL )
        if (self.__model == None):
            log.info('loading detection mode %s',const.DETECTION_MODEL)
            self.__model = load_model(model_file, custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})
            log.info('successfully loaded detection model')
    
    def __setup_data(self): 
        #rear_bounding_boxes = util.load_annotations(const.DATA_PATH+const.BOXES_FOLDER+const.REAR_BOUNDING_BOXES_FILE)
        #front_bounding_boxes = util.load_annotations(const.DATA_PATH+const.BOXES_FOLDER+ const.FRONT_BOUNDING_BOXES_FILE)
        #bounding_boxes = front_bounding_boxes.append(rear_bounding_boxes,ignore_index=True)
        bounding_boxes = util.load_annotations(const.DATA_PATH+const.BOXES_FOLDER+const.BOUNDING_BOXES)
        image_files = util.load_image_files(const.DATA_PATH+const.IMAGES_FOLDER)
        #print('Total number of image files is:', len(image_files))
        #print('Total number of annotations is:', len(bounding_boxes))
        bounding_boxes['Path'] = const.DATA_PATH+const.IMAGES_FOLDER+bounding_boxes['Filename']
        bounding_boxes[['x', 'y', 'w', 'h']].describe()
    
        #image_height1 = 1048
        image_height = 800
        image_width = 2560
        #front_bounding_boxes['x'] = front_bounding_boxes['x']/image_width
        #front_bounding_boxes['w'] = front_bounding_boxes['w']/image_width
        #front_bounding_boxes['y'] = front_bounding_boxes['y']/image_height1
        #front_bounding_boxes['h'] = front_bounding_boxes['h']/image_height1
        #rear_bounding_boxes['x'] = rear_bounding_boxes['x']/image_width
        #rear_bounding_boxes['w'] = rear_bounding_boxes['w']/image_width
        #rear_bounding_boxes['y'] = rear_bounding_boxes['y']/image_height2
        #rear_bounding_boxes['h'] = rear_bounding_boxes['h']/image_height2
        normalized_bboxes = bounding_boxes.copy()
        normalized_bboxes['x'] = normalized_bboxes['x']/image_width
        normalized_bboxes['w'] = normalized_bboxes['w']/image_width
        normalized_bboxes['y'] = normalized_bboxes['y']/image_height
        normalized_bboxes['h'] = normalized_bboxes['h']/image_height

        #normalized_bboxes = front_bounding_boxes.append(rear_bounding_boxes,ignore_index=True)
        normalized_bboxes['Path'] = const.DATA_PATH+ const.IMAGES_FOLDER+normalized_bboxes['Filename']
        normalized_bboxes.head(5)
        self.__min_width = bounding_boxes['w'].min()/image_width
        self.__min_height = bounding_boxes['h'].min()/image_height
       
        #self.__min_width = bounding_boxes['w'].min()/image_width      
        #rear_min_height = rear_bounding_boxes['h'].min()
        #front_min_height = front_bounding_boxes['h'].min()
        #if rear_min_height < front_min_height:
         #   self.__min_height = rear_min_height
        #else:
         #   self.__min_height = front_min_height

         #Set up the image filenames and bounding box labels in np arrays
        X_data = np.asarray(normalized_bboxes['Path'])
        y_data = util.get_labels(normalized_bboxes)
        #Shuffle the data prior to carving off the test split
        X_shuffled, y_shuffled = shuffle(X_data, y_data)
        #Mark the index where the data needs to be split to extract the test split
        test_split_index = int(len(X_shuffled)* self.__TEST_SPLIT)
        #Split the X and y arrays after shuffling
        X = np.split(X_shuffled, [test_split_index, len(X_shuffled)])
        y = np.split(y_shuffled, [test_split_index, len(y_shuffled)])
        X_test = X[0]
        X_split = X[1]
        y_test = y[0]
        y_split = y[1]
        X_train, X_val, y_train, y_val = train_test_split(X_split, y_split, test_size=self.__VAL_SPLIT, random_state=42)
        self.__data_dictionary = {"train":(X_train,y_train),"val":(X_val,y_val),"test":(X_test,y_test)}
        
    def extract_plates(self,images): 
        plates_list = []
        missed_images = []
        plate_found = 0

        #images_files = util.load_image_files(images)
        log.info('number of images are %d',len(images))

        #log_files_path = ''.join([file + ' |' for file in images_files])          
        #log.info('images are = {}'.format(log_files_path))

        #inmemory_images = util.load_images_in_memory(images_files,self.__SCALE_FACTOR,self.__CHANNELS,self.__load_size)
        inmemory_images = util.detector_base64_to_images([image["Base64"] for image in images],self.__SCALE_FACTOR,self.__CHANNELS,self.__load_size)
        predictions = self.__model.predict(inmemory_images)    
        if len(predictions) == 0:
            return (None,0)

        pred_boxes = util.process_predictions(predictions, self.__min_height, self.__min_width)
        log.info('number of predictions are %d',predictions.shape[0])        
        #if(len(images_files)!= predictions.shape[0]):
         #   log.warn('number of images and predictions are not equal')
            
        #Save all plates
        for file, box in tqdm(zip(images, pred_boxes), total = len(images)):
            plate_info = {'TransactionID':0,'PlateNumber':'','Color':'','Country':'','Emirate':'','Category':'','PredictedCategory':-1,'ConfidenceLevel':0,'Status':'','Plate':'',"ImageName":'','PlateCoordinates':'','FullImage':''}            
            #tempfile =  os.path.basename(file)
            tempfile =  file["Filename"]
            plate_info["TransactionID"] = int(tempfile[0:tempfile.find(".")])
            plate_info["ImageName"] = tempfile
                        

            if box.any() != 0.0:
                #sample_image = util.equalize_RGB(util.load_image(file, self.__load_size))
                img = util.base64_to_image(file["Base64"],self.__load_size)
                img = cv2.resize(img, self.__load_size, interpolation=cv2.INTER_AREA)
                sample_image = util.equalize_RGB(img)
                bounding_box = util.get_scaled_bbox(box, self.__load_size)
                plate_image = util.extract_plate(sample_image, bounding_box)
                plate_info["FullImage"] = sample_image
                #cv2.imwrite(destination_folder+tempfile, plate_image)
                image_str = util.image_to_base64(plate_image)
                plate_info["Plate"] =  image_str
                plate_info["PlateCoordinates"] = (bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3])
                plates_list.append(plate_info)
                plate_found+= 1
            else:    
                plate_info["Status"] = "Plate Not Found"                
                plates_list.append(plate_info)                
                missed_images.append(file)        

        #if(len(missed_images) > 0):              
         #     log.warn('total images= %d. images missed= %d list=  %s',len(images_files), len(missed_images), ','.join(missed_images))

        log.info('Done extracting plates')
              
        return (plates_list,plate_found)

 