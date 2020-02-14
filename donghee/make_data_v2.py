from typing import Tuple
import numpy as np
import data_util
import video_util
import audio_util
import face_util
 
def makeFilelist(train_list:list, json_list:list, video_num:int) -> Tuple[np.ndarray, dict]:
    ''' Create file list of the folder '''
    video_json = {}
    for json_path in json_list:
        video_json.update(data_util.readJson(json_path))
        
    _split_count = len(train_list) // video_num
    split_array = np.array_split(np.array(train_list[1:]), _split_count)
    
    return split_array , video_json



def splitArray(a_split:np.ndarray, video_json:dict) -> Tuple[list, list, list]:
    ''' get video --> frame '''
    frame_list = []
    audio_clip_list = []
    video_label_list = []
    
    for video_path in a_split:
        video_data = video_util.getVideo(video_path[0])
        video_nm = video_path[0].split('/')[-1]
        video_label = video_json[video_nm]['label']
        
        _frame_list, _audio_clip_list = video_util.getFramesArray(video_data)
        
        frame_list.extend(_frame_list)
        audio_clip_list.extend(_audio_clip_list)
        video_label_list.append(video_label)
        
    return frame_list , audio_clip_list, video_label_list




def makeDataList(frame_list:np.ndarray, \
                 audio_clip_list:np.ndarray, \
                 video_label:np.ndarray, \
                 face_shape:tuple, \
                 lip_shape:tuple) -> Tuple[list, list]:
    face_list = []
    lip_list = []
    mfcc_list = []
    label_list = []
    for frame, audio_clip in zip(frame_list, audio_clip_list):
        try:
            audio_downsample = audio_util.runDownsampling(audio_clip)
            audio_array = audio_util.getAudioArray(audio_downsample)
            audio_mfcc = audio_util.getMfcc(audio_array)

            face, lip = face_util.getFaceData(frame)
            
            if len(face) < 1:
                continue
            elif len(face) > 1:
                pass
                
            if len(lip) < 1:
                continue
            if len(lip[0]) < 1:
                continue
                

            if len(face) != len(lip):
                continue
                
            #face = video_util.runResize(face,(face_shape[0], face_shape[1]))
            #lip = video_util.runResize(lip,(lip_shape[0], lip_shape[1]))
            
            for a_face, a_lip in zip(face, lip):
                if len(a_face) < 1 or len(a_lip) < 1:
                    break
                face_list.append(a_face)
                lip_list.append(a_lip[0])
                mfcc_list.append(audio_mfcc)
                #data_list.append((a_face, a_lip, audio_mfcc))
                label_list.append(video_label)
        except AttributeError:
            #print('[ERROR] can\'t convert from audio clip to array')
            pass
        
    print('video_label_count',len(label_list))
    
    return face_list, lip_list, mfcc_list, label_list

