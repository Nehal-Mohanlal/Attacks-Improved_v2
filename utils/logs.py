import logging 
import os 
import sys 

def logger_module(logger_name =__name__, logs_folder="logs/adversarial.log"): 
    directory = os.path.dirname(logs_folder) 
    if directory and not os.path.exists(directory): 
        os.makedirs(directory)
        
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO) 
    
    for h in logger.handlers[:]: 
        h.close() 
        logger.removeHandler(h)
    
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    handler = logging.FileHandler(logs_folder, mode='w', encoding='utf-8') 
    handler.setFormatter(formatter) 
    logger.addHandler(handler) 
    
    # stream_handler = logging.StreamHandler(sys.stdout) 
    # stream_handler.setFormatter(formatter) 
    # logger.addHandler(stream_handler)
    
    logger.propagate = False 
    
    return logger 





