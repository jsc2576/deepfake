import face_recognition
import numpy as np
import cv2
from typing import Tuple

def getFaceData(frame:np.ndarray) -> list:
    """
        get face, lip images in a frame
        
        argument
            - frame : a frame in video
            
        return
            - face : faces numpy array list
            - lip : lips numpy array list
            
    """
    _faces = getFace(frame)
    _lips = [getLip(_face) for _face in _faces]
    
    if len(_faces) != len(_lips):
        print("[Warning] don't match faces count and lips count")
        
    return _faces, _lips

    
def getFace(frame:np.ndarray) -> list:
    """
        get face image array by face_locations function
        
        argument
            - frame : a frame numpy array in video
            
        return
            - face images : face images in a frame
    """
    assert frame.ndim == 2 or frame.ndim == 3, "frame is not image (frame dimention is only 2 or 3)"
    
    _bboxs = face_recognition.face_locations(frame, number_of_times_to_upsample=2, model="cnn")
    _faces = []
    
    for _bbox in _bboxs:
        top, right, bottom, left = _bbox
        _faces.append(frame[top:bottom, left:right])
        
    return _faces
    


def getLip(frame:np.ndarray) -> list:
    """
        get lips in face image array by using face_landmarks function
        
        argument
            - frame : face image numpy array
        
        return
            - lip list : lip images in face image
    """
    assert frame.ndim == 2 or frame.ndim == 3, "Dimention of face image is only 2 or 3"
    _face_landmarks_list = face_recognition.face_landmarks(frame)
    
    _lip_list = []
        
    for _face_landmark in _face_landmarks_list:
        if 'top_lip' in _face_landmark: # check exist top lip
            _pts_top = _face_landmark['top_lip']
            _pts_top_append = np.append(_pts_top, [_pts_top[0]], axis=0)

        if 'bottom_lip' in _face_landmark: # check exist bottom lip
            _pts_bottom = _face_landmark['bottom_lip']
            _pts_bottom_append = np.append(_pts_bottom, [_pts_bottom[0]], axis=0)
        
        
        
        if 'top_lip' in _face_landmark and 'bottom_lip' in _face_landmark: # exist all lips
            _pts = np.append(_pts_top_append, _pts_bottom_append, axis=0)
            
        elif 'top_lip' in _face_landmark: # only exist top lip
            _pts = _pts_top_append
            
        elif 'bottom_lip' in _face_landmark: # only exist bottom lip
            _pts = _pts_bottom_append
            
        else: # don't exist all lips
            _lip_list.append([])
            continue
            
            
        _x,_y,_w,_h = cv2.boundingRect(_pts)
        _croped = frame[_y:_y+_h, _x:_x+_w].copy()

        _pts = _pts - _pts.min(axis=0)

        _mask = np.zeros(_croped.shape[:2], np.uint8)
        cv2.drawContours(_mask, [_pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        _dst = cv2.bitwise_and(_croped, _croped, mask=_mask)
        
        _lip_list.append(_dst)
        

    if len(_lip_list) > 1:
        print("[Warning] detect many lips in one face image")

    return _lip_list


