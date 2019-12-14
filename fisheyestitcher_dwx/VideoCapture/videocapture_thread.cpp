// two thread dual fisheye image captureing.
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <pthread.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

void thread_devide(cv::Mat* g_matFrame, int cameraId)
{

	VideoCapture capture;
	capture.open(cameraId);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 1600);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 1200);
	capture.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
	while (true)
	{
		capture >> *g_matFrame;
	}
}

int main( int argc, char** argv )
{	
	// initialize outputpath intersectionId positionId
	string outputpath;
	string outputpath_root;
	int intersectionId;
	int positionId;

	// get the argument
	if (argc == 1)
    {
        cout << "Wrong arguments\n" << "-------------------------\n";
        cout << "please add the arguments \n" 
        	<< "--DIR \n"
        	<< "--intersectionId \n"
        	<< "--positionId \n";
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
    	if (string(argv[i]) == "--dir")
    	{
    		outputpath_root = argv[i + 1];
    		i++;
    	}
        else if (string(argv[i]) == "--intersectionId")
        {
            intersectionId = (float)atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--positionId")
        {
            positionId = (float)atof(argv[i + 1]);
            i++;
        }
        else
            ;
    }
    // if the dir is not existed, then make it.
    if (access(outputpath_root.c_str(), 0) == -1)
    {
    	cout << "----------------------------------\n";
    	cout << "the path " << outputpath_root << " is not existing\n" << "now make it\n";
    	int flag = mkdir(outputpath_root.c_str(), 0777);
    	if (flag == 0) 
    	{
    		cout << "make dir successfully\n";
    	}
    	cout << "----------------------------------\n";
    }
    else
    {
    	cout << "----------------------------------\n";
    	cout << "the path " << outputpath_root << " have existed \n";
    	cout << "----------------------------------\n";
    }

    // get the outputpath and name. 
    outputpath = outputpath_root + "/intersection-" + to_string(intersectionId) + "-" + to_string(positionId) + ".avi";


	// videocapture
	Mat frame1, frame2, ROIImage1, ROIImage2, mask1, mask2, gray_Image1, gray_Image2,  CutFrame1,  CutFrame2;
	int cameraId1 = 0;
	int cameraId2 = 1;
	Mat frame3(1140,2280,CV_8UC3,Scalar(0,0,0));
	// two thread
	thread thread_cam1(thread_devide, &frame1, cameraId1);
	thread thread_cam2(thread_devide, &frame2, cameraId2);
    sleep(2);

	if (frame1.empty())
	{
		cout << "Cam1 is not open!" << endl;
		return -1;
	}
	if (frame2.empty())
	{
		cout << "Cam2 is not open!" << endl;
		return -1;
	}

	// videowriter
	VideoWriter outputvideo;
	outputvideo.open(outputpath, CV_FOURCC('M', 'J', 'P', 'G'), 30.0, Size(2280, 1140), true);

	// image resize
    while (true)
	{
		//capture
		//cut and resize
        CutFrame1 = frame1(Rect(255,46,frame1.cols-460 , frame1.rows-60)); //(1140*1140)
        CutFrame2 = frame2(Rect(294,57,frame2.cols-460 , frame2.rows-60)); //(1140*1140)

        ROIImage1 = frame3(Rect(0,0,1140,1140));
		ROIImage2 = frame3(Rect(1140, 0, 1140, 1140));

		cvtColor(CutFrame1, gray_Image1,CV_RGB2GRAY);
		cvtColor(CutFrame2, gray_Image2, CV_RGB2GRAY);
		mask1 = gray_Image1;
		mask2 = gray_Image2;
		CutFrame1.copyTo(ROIImage1, mask1);
		CutFrame2.copyTo(ROIImage2, mask2);

		//output and write
		outputvideo.write(frame3);
		//show
		imshow("frame3", frame3);

		// if key get pressed, break
		if (waitKey(1) == 27)
		{
			break;
		}
    }
	// thread_cam1.join();
	// thread_cam2.join();
    outputvideo.release();
    return 0;
}


