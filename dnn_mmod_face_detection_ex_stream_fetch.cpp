/*
	Author : SAMARTH TANDON
    DATE   : APRIL -17
    VERSION: 1.7
    The code is a standalone CNN based face detector which has the functionality 
    to extract faces from the video sequences or from multiple videos if a single folder is provided.
    The core part of face extraction is similar to dnn_mmod_face_detection_ex.cpp , the only change
    is the code captures the frame from video files having formats mp4 , mkv and mpeg. 

    The code requires the pre trained model i.e. mmod_human_face_detector.dat file can be 
    downloaded from "http://dlib.net/files/mmod_human_face_detector.dat.bz2".The code can be used for 
    processing batch of frames . The batch size will depend on the RAM of the GPU and resolution of 
    the video .Generally keep the value below 10. The face extracted from the frames will be saved in
    the current Directory with video name and the current time stamp.

    The area to be cropped can be increased by setting the MARGIN.You can also tweek with the value of 
    pyramid_down.
	To run the code ./dnn_mmod_face_detection_ex_stream_fetch <Path: Absolute Folder path / Video path> <Batch Size>
        
*/
#include <iostream>
#include <dlib/dnn.h>
#include <dlib/opencv/cv_image_abstract.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <ctime>

#include <opencv2/highgui.hpp>
#include <opencv/cv.h>
#include <string>
using namespace std;
using namespace dlib;
using namespace cv;
int face_extract(std::vector<matrix<bgr_pixel>> imgs , int per_cent_increase , int img_cols , int img_rows , string Folder_Path);
// ----------------------------------------------------------------------------------------

template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<4>>>>>>>>;

// ----------------------------------------------------------------------------------------
//const char dNN_path[100] = "/home/cdac/workspace/Dlib_face_detector/mmod_human_face_detector.dat";
const char dNN_path[500] = "/home/tapas/dlib-19.2_3/mmod_human_face_detector.dat";
net_type net;
 
#define MARGIN 10         // the Margin value(in %) with which to increase the box size


std::string Make_folder_wd_timeStamp(std::string Dest_path, std::string file_name)
{  
	time_t rawtime;
	struct tm *timeinfo;
	time (&rawtime);
  	timeinfo = localtime (&rawtime);
  	//printf ("Current local time and date: %s", asctime(timeinfo));

  	std::string Cur_time = asctime(timeinfo);
  	Cur_time.erase(Cur_time.begin(),Cur_time.end()-14);
  	Cur_time.erase(Cur_time.begin()+8,Cur_time.end());
  	//string firstName = file_name.substr(0, file_name.find("."));

    string folder_name =Dest_path+ file_name.substr(0, file_name.find(".")) +"_"+Cur_time;
    mkdir((folder_name).c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    return folder_name;
}


int face_detect_video(string filename, int batch_size)
{   
	//string Folder_Path = "/home/cdac/Desktop/Frames+video/Videos_test/video_5/";
	int img_cols=0;
	int img_rows=0;
    
    int per_cent_increase = MARGIN;
    per_cent_increase =per_cent_increase / 2;

    cv::Mat3b frame; 
	std::vector<matrix<bgr_pixel>> imgs;   // RGB compatible with Open CV
	matrix<bgr_pixel> img;

    cv::VideoCapture capture(filename);
    if( !capture.isOpened() )
        throw "Error when reading from file";
    else 
    	{
    		int fps_video = capture.get(CV_CAP_PROP_FPS);

			//fps_video = capture.set(CV_CAP_PROP_FPS, 2);
    		int  number_of_frames = int(capture.get(CV_CAP_PROP_FRAME_COUNT));
    		
    		img_cols  = int (capture.get(CV_CAP_PROP_FRAME_WIDTH));
    		img_rows  = int (capture.get(CV_CAP_PROP_FRAME_HEIGHT));
    		std::string fps_str = "Frames Per second:"+ std::to_string(fps_video);
        	std::cout<<"Video opened!"<<'\n'<<"Video FPS: " << fps_video<<'\t'<<"Video Frames: " <<number_of_frames<<'\n';
        	std::string Folder_Path = Make_folder_wd_timeStamp("",filename);


		    while(1)
		    	{   
			        capture>> frame;
					capture.read(frame);
					
					int img_cols = frame .cols ;
					int img_rows = frame .rows ;

			        int  frame_position = int(capture.get(CV_CAP_PROP_POS_FRAMES));
			       
					std::cout<<"Current frame position: "<<frame_position<< "in total frames "<<number_of_frames<<endl;
					
	        		cv::putText(frame, 
	            		fps_str,
	            		cv::Point(50,50), // Coordinates
	            		cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
	            		1.0, // Scale. 2.0 = 2x bigger
	            		cv::Scalar(255,255,255), // Color
	            		2); // Thickness
		
					if(frame.empty())
						break;

					dlib::assign_image(img, cv_image<bgr_pixel>(frame));
					while(img.size() < 1800*1800)
            			pyramid_up(img);
					imgs.emplace_back(std::move(img));

					if(imgs.size() < batch_size)
					   {
					   	//cout<<imgs.size()<<endl;
					   	continue ;
					   }
					face_extract(imgs , per_cent_increase , img_cols , img_rows , Folder_Path) ;
					
					imgs.clear();
					
				}
			if (imgs.size()>0)
			{
				cout<<"processing left frames "<<imgs.size()<<endl;
				face_extract(imgs , per_cent_increase , img_cols , img_rows , Folder_Path) ;
				imgs.clear();

			}
		}

	return 0;
}

int face_detect_multiple_videos(string folder ,int batch_size) 
{
	int img_cols = 0;
	int img_rows = 0;
    
    int per_cent_increase = MARGIN;
    per_cent_increase =per_cent_increase / 2;
    
    // code to read jpeg files in the folder    
    cv::vector<String> filenames;

    glob(folder, filenames);
  
    //Print All Mp4 files
    std::cout<<"Video files :\n";
    for(int j=0 ; j < filenames.size(); ++j)
    	{	
    		std::string vid_file = filenames[j];
    		vid_file.erase(vid_file.begin(), vid_file.end()-(vid_file.length() - folder.length()) );
    		std::cout<<j+1<<"-"<<vid_file<<endl;
    	}

    net_type net;
    deserialize(dNN_path) >> net;  
    
//  make a standard vector imgs
    std::vector<matrix<bgr_pixel>> imgs;  // RGB image compatible with Open Cv
//  std::vector<matrix<int>>       imgs;  // for gray scale images default size 0
  
    matrix<bgr_pixel> img;       //RGB image
//  matrix<int> img;             // for gray scale image        
    
   	cv::Mat3b frame; 
    
    for (int i = 0; i < filenames.size(); ++i)
    {	
        std::string file_path = filenames[i];
   		cv::VideoCapture capture(file_path);
        
        file_path.erase(file_path.begin(), file_path.end()-(file_path.length() - folder.length()) ) ;
    
    	int fps_video = capture.get(CV_CAP_PROP_FPS);
    	std::string fps_str = "Video Frames Per second:"+ std::to_string(fps_video);

    	int number_of_frames = int(capture.get(CV_CAP_PROP_FRAME_COUNT));
    	img_cols  = int (capture.get(CV_CAP_PROP_FRAME_WIDTH));
    	img_rows  = int (capture.get(CV_CAP_PROP_FRAME_HEIGHT));
   
	    
	    if( !capture.isOpened() )
	        throw "Error when reading from file";
	    else 
	        std::cout<<"\nFile Name:"<<file_path<<'\n'<<"Video FPS: " << fps_video<<'\t'<<"Video Frames: " <<number_of_frames<<'\n'<<"Video opened sucessfully!\n";
    		std::cout<<"Processing in batch_size : "<< batch_size<<endl;
    		std::string Folder_Path = Make_folder_wd_timeStamp(folder,file_path);

    	while(1)
    	{   
	        capture>> frame;
			capture.read(frame);
			
			img_cols = frame.cols ;
			img_rows = frame.rows ;

			if(frame.empty())
				break;
			
			

	        int  frame_position = int(capture.get(CV_CAP_PROP_POS_FRAMES));
	        
			std::cout<<"Current frame position: "<<frame_position<< "\tTotal frames: "<<number_of_frames<<endl;
	        /*
	        cv::putText(frame, 
	            		fps_str,
	            		cv::Point(50,50), // Coordinates
	            		cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
	            		1.0, // Scale. 2.0 = 2x bigger
	            		cv::Scalar(255,255,255), // Color
	            		2); // Thickness
			*/
			dlib::assign_image(img, cv_image<bgr_pixel>(frame));
			//while(img.size()<=1280*720)
			//	{
					
					pyramid_up(img);
					cout<<"Pyramid up "<<img.size()<<endl;
			//	}
			imgs.emplace_back(std::move(img));

			if(imgs.size() < batch_size)
			   {
			   	//cout<<imgs.size()<<endl;
			   	continue ;
			   }
			face_extract(imgs , per_cent_increase , img_cols , img_rows , Folder_Path) ;
			
			imgs.clear();
			
		}
		if (imgs.size()>0)
			{
				cout<<"processing left frames: "<<imgs.size()<<endl;
				face_extract(imgs , per_cent_increase , img_cols , img_rows , Folder_Path) ;
				imgs.clear();
			}
		
   }

	return 0;
}


int face_extract(std::vector<matrix<bgr_pixel>> imgs , int per_cent_increase , int img_cols , int img_rows , string Folder_Path)
{

    static int k =0 ;
	
	char buffer[1000];

	cv::Mat imf,imf2;	
	dlib::rectangle cords;	      
	auto dets = net(imgs);
	int frame_track=0;
	clock_t begin = clock();
	for (auto&& frame:dets)
	{	
		imf = dlib::toMat (imgs[frame_track++]);
		for (auto&& d : frame)
		{	
			cords=d;
			if (cords.width() >=0 || cords.height() >=0 )
				{
					cout << "cords.width: " <<cords.width() << ", cords.height: " << cords.height() << endl;

				int width_margin  =  cords.width() * per_cent_increase/100;
				int height_margin = cords.height() * per_cent_increase/100;
				cout<<"\ncoordinates before adjustment\t"<<cords.left()<<"  "<<cords.top()<<" "<<cords.width()<<" "<<cords.height()<<endl;
				if(cords.left()-width_margin < 0 || cords.top()-height_margin < 0 || cords.width()+ 2*width_margin > img_cols || cords.height()+ 2*height_margin > img_rows )
				   		{	
				   			width_margin  =  0 ; 
				   			height_margin =  0 ;
				   			//std::cout<<"adjustment out of bounds , Margins reset.."<<endl;
				   		}
				cv::Rect roi (cords.left()-width_margin,cords.top()-height_margin, cords.width()+ 2*width_margin, cords.height() + 2*height_margin);
				
				if (roi.x >=0 && roi.y >=0 && roi.width + roi.x < img_cols && roi.height + roi.y < img_rows)	
				{	//cout<<"coordinates after adjustment\t"<<roi<<endl;
					imf2=imf(roi);		
					cv::imshow("Face",imf2);
					cv::waitKey(1);	
					sprintf(buffer,(Folder_Path+"/face_%u.jpg").c_str(),k++);
					imwrite(buffer,imf2);
					cv::rectangle(imf,cvPoint(roi.x,roi.y),cvPoint(roi.x+roi.width,roi.y+roi.height),CV_RGB(255,0,0),4,4);

					int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
	      			double fontScale = 1;
	      			int thickness = 2;  
					cv::putText(imf,std::to_string(cords.width() ),cvPoint(roi.x+roi.width,roi.y), fontFace, fontScale,Scalar::all( 255 ), thickness, 8 );
				}
			}
		}		
	//Size shape(720,480);
	//cv::resize(imf,imf,shape);
	cv::imshow("TEST1",imf);
		cv::waitKey(1);
	}
	imgs.clear();
	clock_t end = clock();
  	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout<<"Frame Processed in :"<< elapsed_secs<<endl;
	return 0;
}


void AssertCond(bool assert_cond, const char* fail_msg) {
  if (!assert_cond) {
    printf("Error: %s\nUsage: ./pic-server <path> <batch_size>\n", fail_msg);
    exit(1);
  }
}

void ParseArgs(int argc, char** argv) {
  AssertCond(argc == 3, "Wrong number of arguments");
  
}

string getFileExt(const string& s) {

   size_t i = s.rfind('.', s.length());
   if (i != string::npos) {
      return(s.substr(i+1, s.length() - i));
   }

   return("");
}

int main(int argc , char ** argv)try
{ 
   
   ParseArgs(argc, argv);
   string Path= argv[1];
   string ext = getFileExt(Path);
   
   int batch_size = atoi(argv[2]);
   
   deserialize(dNN_path) >> net;
   cout<<ext<<endl;  
   if (ext.compare("mkv") == 0 || ext.compare("mp4") == 0 || ext.compare("h265") == 0) 
   		{	cout<<"Processing Video .."<<endl;
   			face_detect_video(Path,batch_size);	
   		}
   else 
		face_detect_multiple_videos(Path+'/',batch_size);
   //face_detect_folder(Path,batch_size);
   //face_detect_rtsp();

  return 0;
}

catch(std::exception& e)
{
    cout << e.what() << endl;
}








/*
int face_detect_folder(string folder , int batch_size ) 
{
    char buffer[100];
    int k=0;
    int per_cent_increase = MARGIN;
    per_cent_increase =per_cent_increase /2 ;
    
    // code to read jpeg files in the folder    
    cv::vector<String> filenames;
    glob(folder, filenames);
    
    int lft_frms=filenames.size()%batch_size;    
	
    net_type net;
    deserialize(dNN_path) >> net;  
    
//  make a standard vector imgs
    std::vector<matrix<bgr_pixel>> imgs;  // RGB image compatible with Open Cv
//  std::vector<matrix<int>>       imgs;  // for gray scale images default size 0
  
    matrix<bgr_pixel> img;       //RGB image
//  matrix<int> img;             // for gray scale image       
    
    for (int i = 0; i < filenames.size(); ++i)
    {	
        std::string file_path = filenames[i];
		std::string Ext = file_path.substr(file_path.find_last_of(".")+1);

		if(Ext == "jpeg" || Ext== "jpg"|| Ext=="png" ) 
    	{	
			load_image(img, file_path);
			int img_cols = frame .cols ;
			int img_rows = frame .rows ;
			cout << "Image shape ["<<img_cols<<"X"<<img_rows<<"]"<<endl;
			// Place images in a batches for batch processing
			imgs.emplace_back(std::move(img));

			if(imgs.size()== lft_frms && i==filenames.size()-1)
			 	batch_size=lft_frms;
			 
			if(imgs.size() < batch_size)
		   		continue ;
			
			
		
		cv::Mat imf,imf2;		      
		auto dets = net(imgs);
		int frame_track=0;
	
		for (auto&& frame:dets)
		  {	
		  	imf = dlib::toMat (imgs[frame_track++]);
		    for (auto&& d : frame)
			{	cords=d;
				cv::Rect roi (cords.left(),cords.top(),cords.width(),cords.height());			
				
				if (roi.x>=0 && roi.y>=0)				
				{	imf2=imf(roi);			
					sprintf(buffer,"/home/cdac/surveillance/face/faces_c++_folder/face_%u.jpg",k++);
					imwrite(buffer,imf2);
				 }
		       }		
			//cv::imshow("TEST",imf);
		    	//cv::waitKey(1);
		  }
		imgs.clear();
		cout<<"Processed"<<endl;
	}
	
   
   }
	return 0;
}
*/












