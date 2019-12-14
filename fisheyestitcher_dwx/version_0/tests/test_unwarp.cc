#include <opencv2/opencv.hpp>
#include "unwarp.h"

int main(int argc, char** argv) {
    cv::Mat src = cv::imread("/home/du_weixin/FisheyePro/fisheye360/data/im0.png");
    cv::Rect rect(80, 0, 480, 480);
    src = cv::Mat(src, rect);
    cv::imshow("show", src);
    cv::waitKey(0);
    EquiRectangular eqt(180.0f, cv::Size(480, 480), 4.0f);
    cv::Mat dst;
    eqt.unwarp(src, &dst);
    cv::imshow("show", dst);
    cv::waitKey(0);
    return 0;
}
