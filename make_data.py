from typing import Tuple
import numpy as np
from util import data_util
from util import video_util
from util import audio_util
from util import face_util


def getFrames(path_list:list, label_list:list) -> Tuple[list, list, list]:
    ''' get video --> frame '''
    frame_list = []
    audio_clip_list = []
    video_label_list = []
    
    for idx, video_path in enumerate(path_list):
        video_data = video_util.getVideo(video_path)
        video_nm = video_path.split('/')[-1]
        video_label = label_list[idx]
        
        _frame_list, _audio_clip_list = video_util.getFramesArray(video_data)
        
        frame_list.extend(_frame_list)
        audio_clip_list.extend(_audio_clip_list)
        video_label_list.extend([video_label] * len(_frame_list))
        
    return frame_list, audio_clip_list, video_label_list




def makeDataList(frame_list:np.ndarray, \
                 audio_clip_list:np.ndarray, \
                 video_label:list) -> Tuple[list, list, list, list]:
    face_list = []
    lip_list = []
    mfcc_list = []
    label_list = []
    for idx, (frame, audio_clip) in enumerate(zip(frame_list, audio_clip_list)):
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
                label_list.append(video_label[idx])
        except AttributeError:
            #print('[ERROR] can\'t convert from audio clip to array')
            pass
        
    print('video_label_count',len(label_list))
    
    return face_list, lip_list, mfcc_list, label_list

