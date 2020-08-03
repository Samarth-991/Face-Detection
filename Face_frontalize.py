import sys
import dlib
import cv2
import openface
import os
predictor_model = "shape_predictor_68_face_landmarks.dat"

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)
def face_align(image):
    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)
    #print("Found {} faces in the image file {}".format(len(detected_faces), file_name))
    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):
        # Detected faces are returned as an object with the coordinates 
        # of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
        # Get the the face's pose
        pose_landmarks = face_pose_predictor(image, face_rect)
        
        # Use openface to calculate and perform the face alignment
        alignedFace = face_aligner.align(500, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        #cv2.imshow('Allied Image',alignedFace)
        #cv2.waitKey(5)
        # Save the aligned image to a file
        return alignedFace

if __name__=='__main__':
    
    imglist = []
    list_L=[]
    relevant_path = "/home/cdac/surveillance/face/faces_c++_folder/"
    included_extenstions = ['jpeg', 'jpg', 'pgm','ppm']
    lst1=os.listdir(relevant_path)
    lst1.sort()
    
    file_names = [fn for fn in lst1
              if any(fn.endswith(ext) for ext in included_extenstions)]
    
    for i in range(len(file_names)):   
        files=relevant_path+file_names[i]
        imglist.append(files)    
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
    
    
    for i in range(len(imglist)):
       
        test_img=cv2.imread(imglist[i])
        #test_img=cv2.resize(test_img, (500,500), interpolation = cv2.INTER_AREA)
        #test_img= cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
        Aligned_image=face_align(test_img)
        cv2.imwrite("/home/cdac/surveillance/face/Fac_aligned/aligned_face_{}.jpg".format(i), Aligned_image)