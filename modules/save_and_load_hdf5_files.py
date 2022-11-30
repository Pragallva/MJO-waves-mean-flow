import h5py
import os
import glob
import numpy as np
from tqdm import tqdm

def make_sure_path_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def save_dict_to_hdf5(dic, filename, track=False):

    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic, track)

def load_dict_from_hdf5(filename, track=False):

    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/', track)


def recursively_save_dict_contents_to_group(h5file, path, dic, track):
    """
    ....
    """
    if track :
        for key, item in tqdm(dic.items()):
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, int, float, list, tuple)):
                h5file[path + key] = item
            elif isinstance(item, dict):
                recursively_save_dict_contents_to_group(h5file, path + key + '/', item, track)
            else:
                raise ValueError('Cannot save %s type'%type(item))
                
    else:
        
        for key, item in dic.items():
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, int, float, list, tuple)):
                h5file[path + key] = item
            elif isinstance(item, dict):
                recursively_save_dict_contents_to_group(h5file, path + key + '/', item, track)
            else:
                raise ValueError('Cannot save %s type'%type(item))
                

def recursively_load_dict_contents_from_group(h5file, path, track): 

    ans = {}
    
    if track :
        for key, item in tqdm(h5file[path].items()):
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/', track)
        return ans 
    
    else:
        for key, item in (h5file[path].items()):
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/', track)
        return ans 

        



                    
        
