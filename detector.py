import sys
import os
import cv2
sys.path.append('../../../facerec-bytefish-cdac/py')
import numpy as np
from facedet.detector import Detector
import dlib
class FaceDetectorWithNose(Detector):
    def __init__(self, cascade_fn_face="haarcascade_frontalface_default", cascade_fn_nose="haarcascade_mcs_nose.xml", 
                 scaleFactorFace=1.3, minNeighborsFace=6, minSizeFace=(30,30), 
                 scaleFactorNose=1.2, minNeighborsNose=4, minSizeNose=(20,20)):
        
        self.face_cascade_xml=cascade_fn_face
        self.nose_cascade_xml=cascade_fn_nose
        
        self.scale_factor_face = scaleFactorFace
        self.min_neighbors_face = minNeighborsFace
        self.flags=0
        self.min_size_face=minSizeFace
        
        self.scale_factor_nose=scaleFactorNose
        self.min_neighbors_nose=minNeighborsNose
        self.flags_nose=0
        self.min_size_nose=minSizeNose
        
    
    def detect(self, src):
        face_cascade = cv2.CascadeClassifier(self.face_cascade_xml)        
        Nose_cascade=cv2.CascadeClassifier(self.nose_cascade_xml)    
          
        face_wd_nose_coords=[]        
        coords = face_cascade.detectMultiScale(src, self.scale_factor_face, self.min_neighbors_face, self.flags, self.min_size_face)
        if len(coords) == 0:
            return np.ndarray((0,))
        
        for (x,y,w,h) in coords: 
            roi_gray = src[y:y+h, x:x+w]
            coords_nose = Nose_cascade.detectMultiScale(roi_gray, self.scale_factor_nose, self.min_neighbors_nose,self.flags_nose, self.min_size_nose)
            if len(coords) == 0:
                return np.ndarray((0,))
            try:
                for (x1,y1,w1,h1) in coords_nose: # x,y as face starting points w,h as width and height for drawing the rectangle
                    #face_wd_nose_coords.append((x,y,w,h))
                    coords[:,2:] += coords[:,:2]
                    return coords
            except: 
                return np.ndarray((0,))
        #return (len(face_wd_nose_coords),coords)
    

class FaceDetectorWithDlib(Detector):
    def __init__(self ):
        detector = dlib.get_frontal_face_detector()
        #predictor = dlib.shape_predictor(self.predictor_path)
        
    def detect(self, src):
    
        dets = self.detector(src,1)
        if len(dets)==0:
            pass
        
        else:
            
            facesCoords=np.array([[d.left(), d.top(),d.right(),d.bottom()] for d in dets])
            faceCoords[:,2:] += faceCoords[:,:2]
            return facesCoords
       