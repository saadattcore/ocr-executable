import logging 
from logging.handlers import TimedRotatingFileHandler
from src.helpers.constants import LOG_FILE

class _FileLogger():  
    def __init__(self):
        self.__logger = None 
        self.__configure()

    def __configure(self):       
        log_format = "%(levelname)s) %(asctime)s: %(message)s"
        data_timestamp_format = "%m/%d/%Y %I:%M:%S %p"
        log_level = logging.DEBUG

        handler = TimedRotatingFileHandler(LOG_FILE,when="midnight",interval=1)
        formatter = logging.Formatter(log_format,data_timestamp_format)        
        handler.setFormatter(formatter)
        handler.suffix = "%Y-%m-%d_%H-%M-%S"
        self.__logger = logging.getLogger("file_logger")
        self.__logger.addHandler(handler)
        self.__logger.setLevel(log_level)

        

    @property
    def logger(self):
        return self.__logger

class LoggerFactory:
    __filelogger = None

    @staticmethod
    def getlogger():
        if LoggerFactory.__filelogger == None:
            LoggerFactory.__filelogger = _FileLogger() 
        return LoggerFactory.__filelogger.logger



  

    
      
    

 
    