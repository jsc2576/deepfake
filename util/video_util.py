from moviepy.video.VideoClip import VideoClip
from moviepy.editor import *
from typing import Tuple, Union
import numpy as np
import audio_util
import cv2



def getVideo(path:str) -> VideoClip:
    """
        get video data
        
        argument
            - path : mp4 file path
            
        return 
            - _video : VideoClip class
    """
    assert os.path.splitext(path)[-1] == ".mp4", "Video Format is not .mp4"
    
    _video = VideoFileClip(path)
    
    return _video
    
    

    
    
def getFramesArray(video:VideoClip, fps:int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
        get all frame arrays by fps
        And, if can't use frame or audio data, filter at try/except AttributeError
        
        argument
            - video : VideoClip class
            - fps : fps getting frames
            
        return
            - _frame_list : all frame array in video
            - _sound_list : all sound array in video
    """
    _frame_list = []
    _sound_list = []
    _i = 0
    
    while _i/fps < video.duration:
        try:
            _frame = video.get_frame(_i/fps)
            _sound = audio_util.getSubAudio(video.audio, start=_i/fps, end=(_i+1)/fps)
            
            _sound_list.append(_sound)
            _frame_list.append(_frame)
        except AttributeError:
            print("[Warning] Error get frame and sound")
            pass
        _i += 1
        
#     _sound_len_min = min([len(_sound) for _sound in _sound_list])
#     _sound_list = [_sound[:_sound_len_min] for _sound in _sound_list]
    
    return np.array(_frame_list), np.array(_sound_list)
    
    
    
    
    
def runResize(frames:Union[np.ndarray, list], size:tuple) -> np.ndarray:
    """
        resize all frame data
        
        argument
            - frames : all frame array in a video
            - size : resize tuple (width, height)
            
        return
            - _frame_resize_list : all frame array after resize
    """
    assert frames.ndim == 3 or (frames.ndim == 4 and frames.shape[3] < 4), "Input frames dimension is 3 or 4"
    
    _frame_resize_list = [cv2.resize(_frame, size, interpolation=cv2.INTER_AREA) for _frame in frames]
    
    return np.array(_frame_resize_list)




if __name__ == "__main__":
    video_data = getVideo('/data/kaggle/teamA/deepfake/data/dfdc_train_part_0/aaqaifqrwn.mp4')
    frames, sounds = getFramesArray(video_data)
    frames_resize = runResize(frames, (32,32))
    
    audios_downsample = [audio_util.runDownsampling(audio_data, sr) for audio_data in sounds]
    
    audio_np = [getAudioArray(_audio_downsample) for _audio_downsample in audios_downsample]
    audio_mfcc = [getMfcc(_audio_np) for _audio_np in audio_np]