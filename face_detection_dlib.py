import dlib
import numpy as np
import cv2
import os
import  pickle
import time
import fileManagement as fm
predictor_path='shape_predictor_68_face_landmarks.dat' 
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

faceShape=(100,100)

#path = "/home/cdac/surveillance/"

def getPaths(path, out_info_file):
    i=0
    #print path
    full_train_dic_name = {} 
    fp=open(out_info_file,'w')
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_dir_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_dir_path):
                # label="{}_{}".format(subject_dir_path.split('/')[-1],i)
                label="{}".format(subject_dir_path.split('/')[-1])
                imgFilePath="{}/{}".format(subject_dir_path,filename)
                full_train_dic_name[imgFilePath] = label
                #print "{} {}".format(imgFilePath, label)
                fp.write("{} {}\n".format(imgFilePath, label))
                # i = i + 1
    #print "full_train_dic_name: {}\n".format(full_train_dic_name)
    pickle.dump( full_train_dic_name, open( "xtrainimagefullsave.p", "wb" ) )
    #print full_train_dic_name
    take_from_subfolders()
    fp.close()
    
def take_from_subfolders():
    
    train_dic_name  = pickle.load( open( "xtrainimagefullsave.p", "rb" ) )
    print train_dic_name
    foldername='folder_'
    faceImagesDir = fm.makefacedir(foldername)
    i = 0000000
    for frame_path in train_dic_name.keys():
        if frame_path.endswith('jpg' or 'jpeg'):
            print 'hello'
            gray_frame = cv2.imread(frame_path)
            no = "{0:0=6d}".format(i)
            faceRecognition(gray_frame,faceImagesDir,no)
            i = i +1
            

def take_fromfolder2(path):
    
    
    foldername='folder_'
    faceImagesDir = fm.makefacedir(foldername)
    i = 0000000
    
    imglist = []
    included_extenstions = ['jpeg', 'jpg', 'png','ppm']
    lst1=os.listdir(path)
    lst1.sort()
    file_names = [fn for fn in lst1 
              if any(fn.endswith(ext) for ext in included_extenstions)]
   
    for j in range(len(file_names)):   
        files=path+file_names[j]
        imglist.append(files)
    print imglist
    
    for k in range(len(imglist)):        
        frame = cv2.imread(imglist[k])
      
        frame=frame[200:960,400:1500]
        frame = cv2.resize(frame,(1100,760), interpolation = cv2.INTER_AREA)
        #gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        no = "{0:0=6d}".format(i)
        faceRecognition(frame,faceImagesDir,no)
        i = i +1
    
    

def faceExtract_dlib(frame):
    start_frame=time.time()
    print type(frame)
    dets = detector(frame,1)
    facesCoords=np.array([[d.left(), d.top(),d.right(),d.bottom()] for d in dets])
    print "Time to process each frame :{}".format(time.time()-start_frame)
    #print facesCoords
    return facesCoords
    

def faceRecognition(gray_frame,faceImagesDir,no):
    
    faceShape = (100,100 )     
    facepic = faceImagesDir+"/face"+str(no)+".jpeg"
    id_name = 'Face'
    distance = '0.0'
    font = cv2.FONT_HERSHEY_SIMPLEX
        
    
    
    facesCoords  = faceExtract_dlib(gray_frame) 
        
    for (x,y,w,h) in facesCoords: # x,y as face starting points w,h as width and height for drawing the rectangle
        if facesCoords  == None :
            pass
             
        cv2.rectangle(gray_frame,(x,y),(w,h),(255,0,0),2)
        candidateFace=gray_frame[y:h,x:w]      
            
        pixel_width=abs(x-w)
        pixel_height=abs(y-h)
        #print pixel_width,pixel_height
        if (pixel_width > 10 and pixel_height >10):
            try:
                P_string=str(pixel_width)+'X'+str(pixel_height)
                cv2.putText(gray_frame, P_string,(w+3,h),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1,color=(0, 0, 255))
                
                candidateFace = cv2.resize(candidateFace, faceShape, interpolation = cv2.INTER_AREA)                
            
                img_path_sans_ext=facepic.split('.')[-2]        
                face = "{}_{}_{}.jpeg".format(img_path_sans_ext,id_name,distance)        
                
                cv2.imwrite(face, candidateFace)
                cv2.imwrite('/tmp/test.jpg',candidateFace)        
                cv2.putText(gray_frame,id_name,(x,y),font, 0.7,(255,255,255),1) 
            except:
                pass
        else:
            pass
        
    cv2.imshow('Stream',gray_frame)
    cv2.waitKey(1)




if __name__=='__main__':
    start=time.time()
    ''' for Hireracy use getpaths code'''
    #getPaths('/home/cdac/Documents/pics/test','/tmp/test.txt')
    ''' for single folder one hirerachy  use / at the end of the path name '''
    take_fromfolder2('/home/cdac/Documents/pics/test/')
    print "Time to process complete data :{}".format(time.time()-start)