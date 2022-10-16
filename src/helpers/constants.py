import os

#MODEL_PATH = '..\\model\\'
#DATA_PATH = '..\\data\\training-plates\\'

MODEL_PATH = 'model\\'
DATA_PATH = 'data\\training-plates\\'
IMAGES_FOLDER = 'images\\'
BOXES_FOLDER = 'boxes\\'
FRONT_BOUNDING_BOXES_FILE = 'front_bounding_boxes.csv'
REAR_BOUNDING_BOXES_FILE = 'rear_bounding_boxes.csv'
BOUNDING_BOXES = 'bounding_boxes_new.csv' 
#BOUNDING_BOXES = 'bounding_boxes.csv' #-- includes bikes

GROUND_TRUTH_FILE = 'ground-truth.csv'
PLATES_FOLDER = 'plates//'
REAR_GROUND_TRUTH_FILENAME = 'rear_groundtruth.csv'
REAR_PLATES = os.path.join(DATA_PATH ,'plates\\rear\\')

TRAINING_PLATES_FOLDER = os.path.join(DATA_PATH , PLATES_FOLDER , 'train\\')
VALIDATION_PLATES_FOLDER = os.path.join(DATA_PATH , PLATES_FOLDER , 'val\\')
TEST_PLATES_FOLDER = os.path.join(DATA_PATH , PLATES_FOLDER , 'test\\')
TOTAL_PLATES_FOLDER = os.path.join(DATA_PATH , PLATES_FOLDER , 'total\\')

OCRX_CLASSIFIED_IMAGES_FOLDER = os.path.join(DATA_PATH,PLATES_FOLDER,'ocrx classified images\\')


# recognition model constants

TRAINING_DATA_PATH = DATA_PATH
TRAINING_IMAGES = 'images\\'
TRAINING_PLATES = 'plates\\'
EXTENSION = '*.jpg'
GT_FILE = 'ground-truth.csv'
DATA_TATTILE = 'ground-truth_Tattile_20171119.csv'
TATTILE_PLATE_PATH = TRAINING_DATA_PATH+'tattile_plates\\'
JAI_PLATE_PATH = TRAINING_DATA_PATH+TRAINING_PLATES+'total\\'
ADD_PLATE_PATH = TRAINING_DATA_PATH+'additional_plates\\'
CARMEN_PLATE_PATH = 'data\\carmen-evaluation\\'


# processed plates paths
#PROCESSED_FOLDER = 'processed images\\'
#PROCESSED_PLATES_FOLDER = os.path.join(PROCESSED_FOLDER, 'plates\\')
#PROCESSED_CLASSIFICATION_FOLDER = os.path.join(PROCESSED_FOLDER, 'classifications\\')
#PROCESSED_OCR_FOLDER = os.path.join(PROCESSED_FOLDER, 'ocr\\')

#SOURCE_FOLDER = os.path.join(DATA_PATH , 'additional_images\\')

#SOURCE_FOLDER = os.path.join(DATA_PATH , 'processed images\\images\\')
PROCESSED_IMAGES_ROOT = 'processed images\\'
SOURCE_FOLDER = ''
PROCESSED_PLATES_FOLDER = ''
MULTILINE_FOLDER = ''   
COLOR_CODE_FOLDER = ''
MULTILINE_PLATE_NUMBER_FOLDER = ''

# models file 

#DETECTION_MODEL = 'plate_detector_Unet_v9.16--0.922.h5' # experimental

DETECTION_MODEL = 'detector model\\plate_detector_Unet_v8.12--0.914.h5' # this is to be used
#DETECTION_MODEL = 'detector model\\plate_detector_Unet_v9.16--0.922.h5' #-- includes bikes detection

#CLASSIFICATION_MODEL = 'plate_classifier_UAT_v01.12-0.074.h5' #!!!!


#CLASSIFICATION_MODEL = 'plate_classifier_v12.12-0.160.h5' #-- including bikes
#CLASSIFICATION_MODEL = 'plate_classifier_v13.19-0.163.h5'
#
#OCR_MODEL = 'OCREngine_UAT_v8.5_BS64_LR1e-3.01-0.153.h5'
#OCR_MODEL = 'OCREngine_v13_BS64_LR1e-3.17-0.242.h5'

#OCR_MODEL = 'OCREngine_UAT_v8.5_BS64_LR1e-3.01-0.153.h5'
CLASSIFICATION_MODEL = 'classifier model\\plate_classifier_Version2_RGB_v04.16-0.055.h5'

OCR_MODEL = 'recognition model\\OCREngine_Emirates_VGG_v9.7_BS64_LR1e-3.19-0.069.h5'

KSA_OM_BIKE_OCR_MODEL = 'recognition model\\OCREngine_UniqueData_BikeSAOM_VGG_v1.1_BS64_LR1e-3.09-0.161.h5'

DUBAI_NT_OCR = 'recognition model\\OCREngine_NT_v3_BS64_LR1e-3.16-0.043.h5'


MULTILINE_OCR = 'recognition model\\OCREngine_AUD_Spacer_v9.5_BS64_LR1e-3.21-0.056.h5'

KSA_OM_CATEGORY_CLASSIFIER = 'oman ksa category classifier\\category-classifier-ksa-om-v1.3.06-0.040.h5'

#MULTILINE_OCR = 'OCREngine_ML_v10_BS64_LR1e-3.13-0.245.h5' #-- including bikes 

#CODE_OCR = 'OCRColor_UAT_v8.5_BS64_LR1e-3.24-0.114.h5' = 


CODE_OCR = 'archieve\\OCRColor_UAT_v8.5_BS64_LR1e-3.09-0.109.h5'

PLATE_COLOR_CLASSIFIER = 'archieve\\dxbnt_aud_shj_classifier_v5.18-0.232.h5'
DUBAI_ML_PLATE_COLOR_CLASSIFIER = 'archieve\\dubai_ntml_code_classifier_v2.27-0.062.h5'



# log file path 
LOG_FILE_PATH = 'logs\\{0}\\'
LOG_FILE = ''