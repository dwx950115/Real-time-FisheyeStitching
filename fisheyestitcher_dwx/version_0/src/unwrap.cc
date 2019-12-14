#include <opencv2/opencv.hpp>
#include <cmath>
#include "unwarp.h"

EquiRectangular::EquiRectangular(
        const float FOV_deg,
        const cv::Size2i src_size,
        const float pix_deg
        ){
    this->FOV_deg = FOV_deg;
    this->src_size = src_size;
    this->dst_size.height = pix_deg * FOV_deg;
    this->dst_size.width  = pix_deg * FOV_deg;
    this->res_pix_deg = pix_deg;
    this->res_pix_rad = pix_deg * 180 / CV_PI;
    this->map_x = cv::Mat(dst_size, CV_32FC1);
    this->map_y = cv::Mat(dst_size, CV_32FC1);
    this->src_radius = src_size.height/2;

    float pitch, yaw, xs, ys;
    for(int yd = 0; yd < dst_size.height; ++yd){
        for(int xd = 0; xd < dst_size.width; ++xd){
            pitch = -(yd +0.5 - dst_size.height/2) / res_pix_rad;
            yaw = (xd +0.5 - dst_size.width/2) / res_pix_rad;
            ys = (1-sin(pitch)) * src_radius;
            xs = (1+cos(pitch)*sin(yaw)) * src_radius;
            this->map_x.at<float>(yd, xd) = xs;
            this->map_y.at<float>(yd, xd) = ys;
        }
    }
}

void EquiRectangular::unwarp(const cv::Mat& src, cv::Mat* dst) {
    cv::remap(src, *dst, this->map_x, this->map_y,
            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
}


