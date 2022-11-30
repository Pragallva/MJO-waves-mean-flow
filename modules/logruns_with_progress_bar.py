import logging
import os

from tqdm import tqdm
import tqdm_logging_wrapper


# class TqdmLoggingHandler(logging.StreamHandler):
#     """Avoid tqdm progress bar interruption by logger's output to console"""
#     # see logging.StreamHandler.eval method:
#     # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
#     # and tqdm.write method:
#     # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

#     def emit(self, record):
#         try:
#             msg = self.format(record)
#             tqdm.write(msg, end=self.terminator)
#         except RecursionError:
#             raise
#         except Exception:
#             self.handleError(record)
            
def make_sure_path_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)  
        
class default_log(object):
    
      def __init__(self, logfilename='log', log_directory='/data/pbarpanda/log/'):
        self.logfilename = logfilename
        self.log_directory  = log_directory
        make_sure_path_exists( path   = log_directory )
        
        if os.path.exists(log_directory+logfilename+'.log'):
            os.remove(log_directory+logfilename+'.log')
        
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(filename = log_directory+logfilename+'.log',level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p') 
            
      def write(self,line):
         logging.debug(line)
        
#       def progress_bar(self, items_iter):
#          with tqdm_logging_wrapper.wrap_logging_for_tqdm(items_iter), item_iter:
#               for item in items_iter:
#                  self.write(line)
        

# p1 = Person("John", 36)
# p1.myfunc()



# import time

# log = logging.getLogger(__name__)
# log.setLevel(logging.INFO)
# log.addHandler(TqdmLoggingHandler())
# #   ^-- Assumes this will be the unique handler emitting messages to sys.stdout.
# #       If other handlers output to sys.stdout (without tqdm.write),
# #       progress bar will be interrupted by those outputs

# for i in tqdm(range(20)):
#     log.info(f"Looping {i}")
#     time.sleep(0.1)