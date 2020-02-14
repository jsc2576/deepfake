from typing import Tuple
import numpy as np
from util import data_util
from util import video_util
from util import audio_util
from util import face_util

def getFramesP(path:str):
    ''' get video --> frame '''
    video_data = video_util.getVideo(path)
    _frame_list, _audio_clip_list = video_util.getFramesArray(video_data)
        
    return _frame_list, _audio_clip_list



def makeDataListP(frame_list:np.ndarray, \
                 audio_clip_list:np.ndarray):
    face_list = []
    lip_list = []
    mfcc_list = []
    
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
                
            for a_face, a_lip in zip(face, lip):
                if len(a_face) < 1 or len(a_lip) < 1:
                    break
                face_list.append(a_face)
                lip_list.append(a_lip[0])
                mfcc_list.append(audio_mfcc)
        except AttributeError:
            pass
    
    return face_list, lip_list, mfcc_list