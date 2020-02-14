from moviepy.editor import *
import numpy as np
from librosa.feature import mfcc
from moviepy.audio.AudioClip import AudioClip
import os

def getAudio(path:str) -> AudioClip:
    """
        only read mp4 file to moviepy audio class
        return AudioClip class
        
        argument
            - path : mp4 video path
            
        return
            - _audio : AudioClip class
    """
    assert os.path.splitext(path)[-1] == ".mp4", "Video Format is not .mp4"
    
    _video = VideoFileClip(path)
    _audio = _video.audio
    
    return _audio





def runDownsampling(audio:AudioClip, sr:int = 8000) -> AudioClip:
    """
        Setted sound rate is smaller than audio sound rate.
        
        argument
            - audio : AudioClip class
            - sr : sound rate that you want
            
        return
            - : downsampling AudioClip class
    """
    assert audio.fps > sr, "Audio sound rate is bigger than sr"
    
    return audio.set_fps(sr)






def getSubAudioArray(audio:AudioClip, start:float = 0, end:float = None) -> np.ndarray:
    """
        get slice audio array
        
        argument
            - audio : AudioClip class
            - start : start time second in audio
            - end : end time second in audio
            
        return
            - _audio_np_mono : only 1 dimention audio data
    """
    
    _audio_sub = getSubAudio(audio)
    _audio_np_mono = getAudioArray(_audio_sub)
    return _audio_np_mono







def getSubAudio(audio:AudioClip, start:float = 0 ,end:float = None) -> AudioClip:
    """
        slice AudioClip
        
        argument
            - audio : AudioClip class
            - start : start time second in audio
            - end : end time second in audio
            
        return 
            - _audio_sub : AudioClip class
    """
    # assert end is not None and start < end, "start time is later than end time"
    
    assert end is None or (end is not None and start < end), "start time is later than end time"
    '''
    shbaek, 200118
    start time is later than end time 에러 메시지가 뜹니다.
    audio.ipynb 기준으로 코드 바꿔서 돌렸습니다.
    '''
    
    _audio_sub = audio.subclip(start, end)
        
    return _audio_sub








def getAudioArray(audio:AudioClip) -> np.ndarray:
    """
        convert from AudioClip to numpy array
        
        argument
            - audio : AudioClip class
            
        return
            - _audio_np_mono : only 1 dimention audio data
    """
    
    _audio_np = audio.to_soundarray()
    
    if _audio_np.ndim == 2:
        _audio_np_mono = np.mean(_audio_np, 1)
    elif _audio_np.ndim == 1:
        _audio_np_mono = _audio_np
    else:
        raise Exception('audio array dimension is only 1 or 2')
            
    assert _audio_np_mono.ndim > 0, "Audio data is empty"
    
    return _audio_np_mono




def getMfcc(audio_np:np.ndarray) -> np.ndarray:
    """
        get mfcc data from audio array
        
        argument
            - audio_np : audio numpy array
            
        return
            - mfcc data (2 dimentions)
    """
    return mfcc(audio_np)





if __name__ == "__main__":
    sr = 8000

    audio_data = getAudio('/data/kaggle/teamA/deepfake/data/dfdc_train_part_0/aaqaifqrwn.mp4')
    audio_downsample = runDownsampling(audio_data, sr)
    
    audio_np = getSubAudioArray(audio_downsample, start=0.5, end=1)
    audio_mfcc = getMfcc(audio_np)    