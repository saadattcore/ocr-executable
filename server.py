import flask
import socket
from flask import request
import base64
import time
import os  
import sys
user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
sys.path.append(user_paths[0] + "/src/")
sys.path.append(user_paths[0] + "/src/core/")
sys.path.append(user_paths[0] + "/src/engines/")
sys.path.append(user_paths[0] + "/src/helpers/")
#print(user_paths)
#print(sys.path)

#import keras
import tensorflow as tf
import keras as K
import src.helpers.constants as c
from src.helpers.utility import clean_up
from flask_cors import CORS
from http import server
import glob
import json
import argparse
import traceback



app = flask.Flask(__name__)
#app.config['PROPAGATE_EXCEPTIONS'] = True
CORS(app)
#host_ip = socket.gethostbyname(socket.getfqdn())
host_ip =socket.gethostbyname(socket.gethostname())
#host_ip = '192.168.1.152'
#processor = anpr()
#processor = None
graph = tf.get_default_graph()
exts = ["jpg","jpeg","png"]
listener_port = -1


@app.route("/ping-server")
def Index():
    return flask.jsonify("Flask Server")

@app.route("/process-plates", methods=["POST"])
def get_plates_info():        
    global processor
    global graph    
    start = time.time()  
    with open('config.json', 'r') as f:
        config = json.load(f)

    env = config["Configurations"]["IsRelease"]
    env = str.lower(env)  
    if env == "true":
        is_release = True
    elif env == "false":
        is_release = False
    else:
        return flask.Response("Invalid IsRelease value in config.json file",status=400,mimetype="application/json")

      

    
    try:
        #if request.method == "POST":
            #clean_up()
        request_data = request.get_json()        
        if len(request_data) > 0:
            images = []
            for request_object in request_data:                 
                file_name = ""
                tran_id = 0
                ext = ""                 
                image = ""

                if is_release:
                    tran_id = request_object["TransactionID"]
                    if tran_id < 1:
                        return flask.Response("{}".format("Invalid transaction id"),status=400,mimetype="application/json")
                    ext = request_object["Ext"]
                    if not ext in exts:
                        return flask.Response("{}".format("Only jpeg,jpg,png extensions are allowed"),status=400,mimetype="application/json")
                    file_name = str(tran_id) + "." + ext
                else:
                    file_name = request_object["ImageName"]
                
                image = request_object["ImageData"] 
                if (image is None or  len(image) == 0) or (file_name is None):
                    continue
                    #return flask.Response("{}".format("Image content cannot be empty"),status=400,mimetype="application/json")

               
                images.append({"Filename":file_name , "Base64":image})
                #image_bytes = base64.b64decode(image)

                #with open(c.SOURCE_FOLDER + file_name ,"wb") as f:
                 #   f.write(image_bytes)      
            with graph.as_default():
                if len(images) > 0:
                    jresult = processor.process_batch(is_release,images)    
                else:
                    return flask.Response("No data found",status=400,mimetype="application/text")

            end = time.time()
            time_taken = round(end - start,2)      
            print(jresult) 
            return flask.jsonify(jresult) 
        else:
            return flask.Response("No data found",status=400,mimetype="application/text")

    except Exception as ex:        
        ex_type , ex_value , ex_traceback  = sys.exc_info() 
        trace = traceback.format_exc() 
        ex_name = ex_type.__name__ 
        ex_error = str(ex_value)        
        msg = "{}|{}".format(ex_name,ex_error)
        log.error(ex)
        log.error(trace)
        return flask.Response(msg,status=500,mimetype="application/json")

@app.route("/get_plates_data", methods = ["GET"])       
def get_plate_images():
    data_list = []    
    files = glob.glob(c.PROCESSED_PLATES_FOLDER  + "*.jpg")
    for file in files:
        data_dictionary = {}
        f_stream = open(file,"rb")      
        f_bytes = f_stream.read()                  
        imag_bytes = base64.b64encode(f_bytes)
        image_str = imag_bytes.decode("utf-8")
        file_name = file.split("\\")[-1]
        data_dictionary["ImageName"] = file_name
        data_dictionary["Plate"] = image_str
        data_list.append(data_dictionary)        
        if not f_stream.closed:
            f_stream.close()
    
    return flask.jsonify(data_list)

def create_data_processing_folders(port):
        #port_folder = c.PROCESSED_IMAGES_ROOT +  str(port) + "\\"
        #c.SOURCE_FOLDER = os.path.join(port_folder, "images\\")
        #c.PROCESSED_PLATES_FOLDER = os.path.join(port_folder + "plates\\")
        #multiline_folder = os.path.join(port_folder + "multiline plates\\")
        #c.COLOR_CODE_FOLDER = os.path.join(multiline_folder + "color code\\")
        #c.MULTILINE_PLATE_NUMBER_FOLDER = os.path.join(multiline_folder + "plates numbers\\")    

        ## also keep log file in seperate folder
        c.LOG_FILE_PATH  = c.LOG_FILE_PATH.format(port)

        #if not os.path.exists(port_folder):
         #   os.mkdir(port_folder)

        #if not os.path.exists(c.SOURCE_FOLDER):
         #   os.mkdir(c.SOURCE_FOLDER)

        #if not os.path.exists(c.PROCESSED_PLATES_FOLDER):
         #   os.mkdir(c.PROCESSED_PLATES_FOLDER)

        #if not os.path.exists(multiline_folder):
         #   os.mkdir(multiline_folder)

        #if not os.path.exists(c.COLOR_CODE_FOLDER):
         #   os.mkdir(c.COLOR_CODE_FOLDER)

        #if not os.path.exists(c.MULTILINE_PLATE_NUMBER_FOLDER):
         #   os.mkdir(c.MULTILINE_PLATE_NUMBER_FOLDER)

        if not os.path.exists(c.LOG_FILE_PATH):
            os.mkdir(c.LOG_FILE_PATH)

        c.LOG_FILE = os.path.join(c.LOG_FILE_PATH,"applogs.log")
    

#if __name__ == "__main__":

with open('config.json', 'r') as f:
        config = json.load(f)
enable_gpu = config["Configurations"]["EnableCUDADevice"] 

if enable_gpu == "false":
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

parser = argparse.ArgumentParser()
parser.add_argument("--port",required=True,help="port where ocr server listen http request")
args = vars(parser.parse_args())
print("User pass port value={} from command line.".format(args["port"]))
port = args["port"]

if port is None:
    port = 5000

create_data_processing_folders(port)
from src.anpr import anpr
processor = anpr()
from src.helpers.imports import log
app.run(host=host_ip,port=port,debug=False)    


