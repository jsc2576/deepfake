from typing import Tuple
import numpy as np
import random
import json
import csv
import os

def readData(path:str, ext=['.mp4']) -> list:
    assert len(path) > 0, "path string is empty"
    
    _data_path = []
    
    for _path, _dir, _files in os.walk(path):
        for _filename in _files:
            _ext = os.path.splitext(_filename)[-1]
            
            if _ext not in ext:
                continue
                
            _data_path.append(os.path.join(_path, _filename))
            
    return _data_path



def readJson(path:str) -> dict:
    with open(path) as json_file:
        json_data = json.load(json_file)
    
    return json_data




def splitData(path_list:list, train:float=0.8, valid:float=0.1) -> Tuple[list, list, list]:
    
    """
        split path data to train, valid, test
        
        if train+valid less than 1, other data is test data
    """
    for i in range(5):
        random.shuffle(path_list)
        
    assert (train + valid) <= 1, "please sum of train, valid is less than 1"
    _path_len = len(path_list)
    
    _train_path_list = path_list[:int(_path_len*train)]
    _valid_path_list = path_list[int(_path_len*train):int(_path_len*(train+valid))]
    
    if train + valid == 1:
        _test_path_list = None
    else:
        _test_path_list = path_list[int(_path_len*(train+valid)):]
    
    return _train_path_list, _valid_path_list, _test_path_list




def saveToCSV(data_list:list, filename:str, header:list=None):
    assert np.array(header).ndim == 1 or header is None, "header dimention is only 1 or None"
    
    with open(filename, 'w', encoding='utf-8') as f:
        _wr = csv.writer(f)
        
        if header is not None:
            _wr.writerow(header)
            
        for _data_row in data_list:
            _wr.writerow([_data_row])
            
            
            
def readCSV(csv_path:str):
    _csv_list = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        _wr = csv.reader(f)
        
        for _row in _wr:
            _csv_list.append(_row)
            
    return _csv_list