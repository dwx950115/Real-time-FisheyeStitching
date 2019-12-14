#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>


using namespace cv;
using namespace std;


int main(int, char**)
{
    Mat frame1, frame2;
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap1;
    VideoCapture cap2;
    // open the default camera using default API
    cap1.open(0);
    cap2.open(1);
    // OR advance usage: select any API backend
    // int deviceID = 0;             // 0 = open default camera
    // int apiID = cv::CAP_ANY;      // 0 = autodetect default API
    // open selected camera using selected API
    // cap1.open(deviceID + apiID);
    // check if we succeeded
    if (!cap1.isOpened()) {
        cerr << "ERROR! Unable to open camera1\n";
        return -1;
    }
    if (!cap2.isOpened()) {
        cerr << "ERROR! Unable to open camera2\n";
        return -1;
    }
    //--- GRAB AND WRITE LOOP
    cout << "Start grabbing" << endl
        << "Press any key to terminate" << endl;
    for (;;)
    {
        // wait for a new frame1 from camera and store it into 'frame1'
        // wait for a new frame2 from camera and store it into 'frame2'
        cap1.read(frame1);
        cap2.read(frame2);
        // check if we succeeded
        if (frame1.empty()) {
            cerr << "ERROR! blank frame1 grabbed\n";
            break;
        }
        if (frame2.empty()) {
            cerr << "ERROR! blank frame2 grabbed\n";
            break;
        }
        // show live and wait for a key with timeout long enough to show images
        imshow("Live1", frame1);
        imshow("Live2", frame2);
        // if key get pressed, break
        if (waitKey(1) == 27)
        {
            break;
        }
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

