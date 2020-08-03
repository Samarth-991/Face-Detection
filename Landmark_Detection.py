import dlib
import numpy as np
import cv2
import math
import operator
import os
#from skimage import io
#from skimage.draw import polygon_perimeter

predictor_path='shape_predictor_68_face_landmarks.dat' 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
'''
FACE_POINTS       = list(range(1, 68))
MOUTH_POINTS      = list(range(48, 68))
RIGHT_BROW_POINTS = list(range(17, 21))
LEFT_BROW_POINTS  = list(range(22, 26))
RIGHT_EYE_POINTS  = list(range(36, 41))
LEFT_EYE_POINTS   = list(range(42, 47))
NOSE_POINTS       = list(range(27, 35))
JAW_POINTS        = list(range(0, 16))
'''
RIGHT_EYE_POINTS  = list(range(36, 42))
LEFT_EYE_POINTS   = list(range(42, 48))
NOSE_POINTS       = list(range(27, 36))
MOUTH_POINTS      = list(range(48, 68))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS  = list(range(22, 27))
RIGHT_EAR_POINTS   = list(range( 1, 2))
LEFT_EAR_POINTS  = list(range(15, 16))

def overlay(img):
    # creating  an overlay graph of the face 
    win = dlib.image_window()
    win.clear_overlay()
    win.set_image(img)
    
    dets = detector(img, 1)
    
    for k, d in enumerate(dets):
      shape = predictor(img, d)  
      #Draw the face landmarks on the screen
      win.add_overlay(shape)
    #draw the bounding box on the detected face.  
    win.add_overlay(dets)
    
def find_mean_point(Point_range,landmarks):
    x=0
    y=0
    for i in range(len(Point_range)):
        x=x+landmarks[Point_range[i],0]
        y=y+landmarks[Point_range[i],1]
    pos=(x/len(Point_range ),y/len(Point_range))
    return pos

def find_coordinate(landmarks,point): 
    x=landmarks[point,0]
    y=landmarks[point,1]
    pos=(x,y)
    return pos


def distance(pos1,pos2,im):
    
    x1,y1=pos1
    x2,y2=pos2
    x=(x1+x2)/2
    y=(y1+y2)/2
    pos=(x,y)

    dist = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
    '''
    cv2.putText(im, str(dist),pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 255, 0))
    '''
    return distance
def draw_line(pos1,pos2,im):
    cv2.line (im,pos1,pos2,color=(255,0,0))


def Face_graph(landmarks,im):
    face_coord={}
    ''' Mid Points of Eyes'''
    R_EYE_point=find_mean_point(RIGHT_EYE_POINTS, landmarks)
    L_EYE_point=find_mean_point(LEFT_EYE_POINTS, landmarks)
    
    '''Nose Points'''
    Nose_Bridge     =find_coordinate(landmarks, 27)
    right_Nostril   =find_coordinate(landmarks, 31)
    left_nostril    =find_coordinate(landmarks, 35)
    Nose_tip        =find_coordinate(landmarks, 30)
    Nose_edge       =find_coordinate(landmarks, 33)
    
    '''JAW Points'''
    l_ear_mid_point   =find_mean_point(LEFT_EAR_POINTS , landmarks)
    r_ear_mid_point  =find_mean_point(RIGHT_EAR_POINTS, landmarks)
    r_jaw_point       =find_coordinate(landmarks, 3)
    r_lower_jaw_point =find_coordinate(landmarks, 5)
    l_jaw_point      =find_coordinate(landmarks, 13)
    l_lower_jaw_point=find_coordinate(landmarks, 11)
    Lower_jaw_point      =find_coordinate(landmarks, 8)
    ''' Left eyebrow points'''
    r_brow_point      =find_coordinate(landmarks, 17)
    r_mid_brow_point  =find_coordinate(landmarks, 19)
    rmost_brow_point  =find_coordinate(landmarks, 21)
    '''Right Eyebrow point'''
    l_brow_point    =find_coordinate(landmarks,26 )
    l_mid_brow_point =find_coordinate(landmarks,24 )
    lmost_brow_point=find_coordinate(landmarks,22 )

    '''mouth points'''
    right_mouth_point   =find_coordinate(landmarks, 48)
    top_mouth_point    =find_coordinate(landmarks, 51)
    left_mouth_point   =find_coordinate(landmarks, 54)
    lower_mouth_point  =find_coordinate(landmarks, 57)
    
    face_coord={'R_EYE_point':R_EYE_point,'L_EYE_point':L_EYE_point,'Nose_Bridge':Nose_Bridge,'right_Nostril':right_Nostril,
                'left_nostril':left_nostril,'Nose_tip':Nose_tip,' Nose_edge': Nose_edge,'l_ear_mid_point':l_ear_mid_point,'r_ear_mid_point':r_ear_mid_point,
                'r_jaw_point':r_jaw_point,'r_lower_jaw_point ':r_lower_jaw_point,'l_lower_jaw_point':l_lower_jaw_point,'l_jaw_point  ':l_jaw_point,' Lower_jaw_point': Lower_jaw_point,
                'r_brow_point': r_brow_point ,'r_mid_brow_point':r_mid_brow_point,' rmost_brow_point': rmost_brow_point,'l_brow_point':l_brow_point,'l_mid_brow_point':l_mid_brow_point,
                'lmost_brow_point':lmost_brow_point,'right_mouth_point':right_mouth_point,'top_mouth_point':top_mouth_point,'left_mouth_point':left_mouth_point,'lower_mouth_point':lower_mouth_point
                }
    #print face_coord.items()
    '''Eyebrows nose eyes connection points'''
    draw_line( R_EYE_point,  Nose_Bridge, im)
    draw_line( L_EYE_point,  Nose_Bridge, im)
    
    draw_line( R_EYE_point,  r_brow_point  , im)
    draw_line( R_EYE_point,  r_mid_brow_point, im)
    draw_line( R_EYE_point, rmost_brow_point, im)
    
    draw_line( L_EYE_point,  l_brow_point  , im)
    draw_line( L_EYE_point,  l_mid_brow_point, im)
    draw_line( L_EYE_point, lmost_brow_point, im)
    
    
    
    draw_line( l_mid_brow_point, lmost_brow_point,im)
    draw_line( l_mid_brow_point, l_brow_point, im)
    
    draw_line( r_mid_brow_point, rmost_brow_point,im)
    draw_line( r_mid_brow_point, r_brow_point, im)
    
    draw_line( rmost_brow_point,  lmost_brow_point  , im)
    
    draw_line( R_EYE_point, r_ear_mid_point, im)
    draw_line( L_EYE_point, l_ear_mid_point, im)
    
    draw_line( r_brow_point, r_ear_mid_point, im)
    draw_line( l_brow_point, l_ear_mid_point, im)
    
    draw_line( R_EYE_point,  right_Nostril, im)
    draw_line( L_EYE_point, left_nostril, im)
    
    draw_line( r_ear_mid_point,  right_Nostril, im)
    draw_line( l_ear_mid_point, left_nostril, im)
    
        
    draw_line( Nose_edge,  right_Nostril, im)
    draw_line( Nose_edge, left_nostril, im)
    
    draw_line( Nose_tip,  right_Nostril, im)
    draw_line( Nose_tip, left_nostril, im)
    
    draw_line( Nose_Bridge,  right_Nostril, im)
    draw_line( Nose_Bridge, left_nostril, im)
    
    draw_line( Nose_tip, Nose_Bridge , im)
    
    '''connecton with jaw points'''
    draw_line( r_jaw_point, r_ear_mid_point, im)
    draw_line( l_jaw_point , l_ear_mid_point, im)
    
    draw_line( r_jaw_point,r_lower_jaw_point , im)
    draw_line( l_jaw_point ,l_lower_jaw_point, im)
    
        
    draw_line( Lower_jaw_point ,r_lower_jaw_point , im)
    draw_line(  Lower_jaw_point  ,l_lower_jaw_point, im)
    
    '''mouth and nose connection '''
    draw_line( left_mouth_point, left_nostril, im)
    draw_line( right_mouth_point,right_Nostril, im)
    
    draw_line( left_mouth_point, top_mouth_point, im)
    draw_line( right_mouth_point,top_mouth_point, im)
    
    draw_line( left_mouth_point, lower_mouth_point, im)
    draw_line( right_mouth_point,lower_mouth_point, im)
    
    draw_line(top_mouth_point  , Nose_edge, im)
    
    '''mouth and jaw connection'''
    draw_line( left_mouth_point, l_lower_jaw_point, im)
    draw_line( right_mouth_point,r_lower_jaw_point, im)
    
    draw_line( lower_mouth_point,Lower_jaw_point , im)
    
    '''nose and jaw'''
    draw_line( left_nostril, l_jaw_point, im)
    draw_line( right_Nostril,r_jaw_point, im)

    
    return im,face_coord
def get_landmarks(img):
    
    dets = detector(img)
   
    if len(dets)==0:
        
        return 
    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #   k, d.left(), d.top(), d.right(), d.bottom()))
        landmarks= np.matrix([[p.x, p.y] for p in predictor(img, dets[k]).parts()])
   
    return landmarks
def Find_angle(point1,point2):
    [x1,y1]=point1
    [x2,y2]=point2
    return math.atan((y2-y1)/(x2-x1))

def annotate_landmarks(im, landmarks):
  
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        '''
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        '''
        cv2.circle(im, pos, 2, color=(0,255, 0))
        
    return im

def face_detection(test_img):
    
    dets = detector(test_img,1)
    print("Number of faces detected: {}".format(len(dets)))
    if len(dets)==0:
        pass
    
    else:
        facesCoords=np.array([[d.left(), d.top(),d.right(),d.bottom()] for d in detector(test_img)])
        for (x,y,w,h) in facesCoords:
            
            #extractedface=test_img[y-30:h+10,x-10:w+28]
            #candidateFace = cv2.resize(extractedface, (200,200) ,interpolation = cv2.INTER_AREA)
           
            landmarks=get_landmarks(test_img)
            test_img=annotate_landmarks(test_img, landmarks)
            if landmarks==None:
                continue
            #test_img,face_cord=Face_graph(landmarks, test_img)
           
            cv2.rectangle(test_img,(x,y-5),(w+15,h+15),(0,0,255),2) 
            '''
            cv2.putText(test_img, string,(0,470),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                            fontScale=0.6,
                            color=(0, 0, 255))
    
            '''
    cv2.imwrite('/home/cdac/Desktop/Temp/image.png',test_img)
    cv2.imshow('Frame',test_img)
    cv2.waitKey(0)
    
if __name__=='__main__':

    '''
    imglist = []
    list_L=[]
    relevant_path = "/home/cdac/surveillance/frame/nocam_2016-07-15_14-49-27/"
    included_extenstions = ['jpeg', 'jpg', 'png']
    lst1=os.listdir(relevant_path)
    lst1.sort()
    #print lst1
    file_names = [fn for fn in lst1
              if any(fn.endswith(ext) for ext in included_extenstions)]
    
    for i in range(len(file_names)):   
        files=relevant_path+file_names[i]
        imglist.append(files)
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
    
    #[L_gabor,O_gabor]=Gabor_feature_generate()

    for i in range(len(imglist)):
       
        test_img=cv2.imread(imglist[i])
        test_img=cv2.resize(test_img, (720,576), interpolation = cv2.INTER_AREA)
     
        face_detection(test_img)
    
    '''
    test_img=cv2.imread('IMG.jpg')
    test_img=cv2.resize(test_img, (720,540), interpolation = cv2.INTER_AREA)
     
    face_detection(test_img)
    