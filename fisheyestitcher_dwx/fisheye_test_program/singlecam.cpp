#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;
int main(){
VideoCapture cap(0);
cap.set(CV_CAP_PROP_FRAME_WIDTH, 1600);
cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1200);
cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
Mat img;
while(1){
cap.read(img);
imshow("singlecam", img);
waitKey(1);
}
return 0;
}
