from util import data_util, video_util, face_util
import json
import random


json_list = data_util.readData('../../data/train_videos', ['.json'])
video_list = data_util.readData('../../data/train_videos', ['.mp4'])

video_path_dict = {}
video_json = {}

real_video_list = []
fake_video_list = []

for path in json_list:
    a_json = data_util.readJson(path)
    video_json.update(a_json)    

for k,v in video_json.items():
    if v['label'] == 'FAKE':
        fake_video_list.append(k)
    else:
        real_video_list.append(k)
        
for path in video_list:
    filename = path.split('/')[-1]
    video_path_dict[filename] = path
    

random.seed(100)
random.shuffle(real_video_list)
random.seed(100) 
random.shuffle(fake_video_list)

train_real_video_list = real_video_list[:int(len(real_video_list)*0.8)]
train_fake_video_list = fake_video_list[:int(len(fake_video_list)*0.8)]

valid_real_video_list = real_video_list[int(len(real_video_list)*0.8):]
valid_fake_video_list = fake_video_list[int(len(fake_video_list)*0.8):]


def getRealCnt():
    return len(train_real_video_list)


def getSamples(start, end):
    if end < len(train_real_video_list):
        sample_real = train_real_video_list[start:end]
    else:
        sample_real = train_real_video_list[start:]
    sample_fake = random.sample(train_fake_video_list, end-start)

    path_real = [video_path_dict[filename] for filename in sample_real]
    path_fake = [video_path_dict[filename] for filename in sample_fake]
    
    return path_real, path_fake


def getValid(video_cnt:int):
    if video_cnt == -1:
        sample_real = valid_real_video_list
        sample_fake = valid_fake_video_list
    
    else:
        #sample_real = random.sample(valid_real_video_list, video_cnt)
        sample_real = valid_real_video_list[:video_cnt]
        #sample_fake = random.sample(valid_fake_video_list, video_cnt)
        sample_fake = valid_fake_video_list[:video_cnt]

    path_real = [video_path_dict[filename] for filename in sample_real]
    path_fake = [video_path_dict[filename] for filename in sample_fake]
    
    return path_real, path_fake