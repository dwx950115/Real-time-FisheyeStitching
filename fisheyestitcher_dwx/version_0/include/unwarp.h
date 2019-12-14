#include <opencv2/opencv.hpp>

class EquiRectangular {
    private:
        cv::Mat map_x, map_y;
        float FOV_deg, src_radius, res_pix_deg, res_pix_rad;
        cv::Size2i src_size, dst_size;
    public:
        void unwarp(const cv::Mat& src, cv::Mat * dst);
        EquiRectangular(
                const float FOV_deg,
                const cv::Size2i src_size,
                const float resolution_pixel_per_deg
                );
};
