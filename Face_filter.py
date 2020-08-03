import numpy as np
import cv2
import dlib
import os
import pickle
import openface
from _mysql import NULL
# Dlib HOG face detector 
predictor_path= 'shape_predictor_68_face_landmarks.dat' 
detector      = dlib.get_frontal_face_detector()
predictor     = dlib.shape_predictor(predictor_path)
face_aligner = openface.AlignDlib(predictor_path)
#save_path     ='/home/cdac/Documents/pics/alinged_faces/'

def Make_folder(path):
    if not os.path.isdir(path+'/Filtered_faces'):
        os.makedirs(path+'/Filtered_faces')
    
    save_path = path+'/Filtered_faces/'
    return save_path 
    

def validate_path(PATH):
    if (os.path.isdir(PATH))and os.access(PATH, os.R_OK) == True:
        print "Path exists and accessable"
    else:
        print "Path missing or Unreadable"
        sys.exit(errno.EACCES)

def getPaths(path, out_info_file):
    validate_path(path)
    
    full_train_dic_name = {} 
    
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_dir_path = os.path.join(dirname, subdirname)
            
            for filename in os.listdir(subject_dir_path):
               
                label="{}".format(subject_dir_path.split('/')[-1])
                imgFilePath="{}/{}".format(subject_dir_path,filename)
                
                full_train_dic_name[imgFilePath] = label
                #print "{} {}".format(imgFilePath, label)
                fp.write("{} {}\n".format(imgFilePath, label))
    pickle.dump( full_train_dic_name, open( "xtrainimagefullsave.p", "wb" ) )
    
def get_subdirectory(folder_path):
    validate_path(folder_path)
    sub_directories_list = []
    for dirname, dirnames, filenames in os.walk(folder_path):
        for subdirname in dirnames:
            subject_dir_path = os.path.join(dirname, subdirname)
            sub_directories_list.append(subject_dir_path+'/')
    print sub_directories_list  
    getimg_frm_folder(sub_directories_list)
    
def getimg_frm_folder(sub_dir_list):
    for sub_dir_path in sub_dir_list:
        imglist = []
        included_extenstions = ['jpeg', 'jpg', 'png','pgm','tif']
        lst1=os.listdir(sub_dir_path)
        lst1.sort()
        file_names = [fn for fn in lst1 
                  if any(fn.endswith(ext) for ext in included_extenstions)]
       
        for j in range(len(file_names)):   
            files=sub_dir_path+file_names[j]
            imglist.append(files)
       
        #pickle.dump( imglist, open( "List_Images.p", "wb" ) )
        
        extract_image(imglist ,sub_dir_path)

  
def extract_image(image_list,sub_dir_path):
    save_path = Make_folder(sub_dir_path)
    #train_dic_name   = pickle.load( open( "xtrainimagefullsave.p", "rb" ) )
    #image_list       = pickle.load(open("List_Images.p","rb"))
    img_index =0
    
    #for image_path in train_dic_name.keys():
    for image_path in image_list:  
        print "Processing file : {}".format(image_path)
        img = cv2.imread(image_path)
        face_detection_filter(img,img_index,save_path)
        img_index = img_index +1

def face_detection_filter(face_img,img_index,save_path):
    dets = detector(face_img,1)
  
    if dets == 0: 
        pass
    else :
        for k, d in enumerate(dets):
            landmarks= np.matrix([[p.x, p.y] for p in predictor(face_img, dets[k]).parts()])
            if not (landmarks is None):
                # Use openface to calculate and perform the face alignment
                alignedFace = face_aligner.align(face_img.shape[1], face_img, d, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                cv2.imwrite(save_path+str(img_index)+".jpg",alignedFace)
            


if __name__=='__main__':
   
    path = "/home/cdac/Documents/pics/all_faces_30012016/"
    #getPaths(path,'/tmp/test.txt')
    get_subdirectory(path)
 