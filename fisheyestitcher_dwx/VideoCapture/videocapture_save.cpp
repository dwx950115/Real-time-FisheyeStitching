#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <pthread.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;



int main( int argc, char** argv )
{	
	// defort outputpath
	string outputpath = "./frame/video_test.avi";

	// get the argument
	if (argc == 1)
    {
        cout << "Wrong arguments\n";
        return -1;
    }

    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--op")
        {
            outputpath = argv[i + 1];
            i++;
        }
        else
            ;
    }

	// videocapture
	Mat frame1, frame2,ROIImage1, ROIImage2,mask1,mask2,gray_Image1, gray_Image2,  CutFrame1,  CutFrame2;
	Mat frame3(1140,2280,CV_8UC3,Scalar(0,0,0));

	VideoCapture capture1(0);
	VideoCapture capture2(1);
	capture1.set(CV_CAP_PROP_FRAME_WIDTH, 1600);
	capture1.set(CV_CAP_PROP_FRAME_HEIGHT, 1200);
	capture2.set(CV_CAP_PROP_FRAME_WIDTH, 1600);
	capture2.set(CV_CAP_PROP_FRAME_HEIGHT, 1200);
	capture1.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
	capture2.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
	if (!capture1.isOpened())
	{
		cout << "Cam1 is not open!" << endl;
		return -1;
	}
	if (!capture2.isOpened())
	{
		cout << "Cam2 is not open!" << endl;
		return -1;
	}

	// videowriter
	VideoWriter outputvideo;
	outputvideo.open(outputpath, CV_FOURCC('M', 'J', 'P', 'G'), 15.0, Size(2280, 1140), true);

	// image resize
    while (true)
	{
		//capture
		capture1.grab();
		capture2.grab();
		capture1.retrieve(frame1);
		capture2.retrieve(frame2);
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
		waitKey(1);

		if (char(waitKey(1))=='q') break;

    }

    outputvideo.release();
    return 0;
}




