#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp> // for findHomograph


using namespace std;
using namespace cv;


#define     MAX_FOVD      195.0f

#ifndef _MY_DEBUG_
#   define    _MY_DEBUG_     0
#endif

#ifndef _PROFILING_
#   define    _PROFILING_    0
#endif

//  多项式系数
#define    P1_    -7.5625e-17
#define    P2_     1.9589e-13
#define    P3_    -1.8547e-10
#define    P4_     6.1997e-08
#define    P5_    -6.9432e-05
#define    P6_     0.9976

/***************************************************************************************************
*
* Fisheye Unwarping 鱼眼展开
*   Unwarp source fisheye -> 360x180 equirectangular
*
***************************************************************************************************/
void 
fish_unwarp( const cv::Mat &map_x, const cv::Mat &map_y, const cv::Mat &src, cv::Mat &dst )
{
    remap(src, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
} /* fish_unwarp() */


/***************************************************************************************************
*
* Function: convert fisheye-vertical to equirectangular 
*   (reference: Panotool)
*
***************************************************************************************************/
void 
fish2Eqt( double x_dest, double  y_dest, double* x_src, double* y_src, double W_rad )
{
    double phi, theta, r, s;
    double v[3];
    phi   = x_dest / W_rad;
    theta = -y_dest / W_rad + CV_PI / 2;

    if (theta < 0)
    {
        theta = -theta;
        phi += CV_PI;
    }
    if (theta > CV_PI)
    {
        theta = CV_PI - (theta - CV_PI);
        phi += CV_PI;
    }

    s = sin(theta);
    v[0] = s * sin(phi);
    v[1] = cos(theta);
    r = sqrt(v[1] * v[1] + v[0] * v[0]);
    theta = W_rad * atan2(r, s * cos(phi));
    //
    *x_src = theta * v[0] / r;
    *y_src = theta * v[1] / r;

} /* fish2Eqt() */


/***************************************************************************************************
*
* Map 2D fisheye image to 2D projected sphere
*
***************************************************************************************************/
void 
fish_2D_map( cv::Mat &map_x, cv::Mat &map_y, int Hs, int Ws, int Hd, int Wd, float in_fovd )
{
    static double W_rad = Wd / (2 * CV_PI);
    double x_d, y_d; // dest
    double x_s, y_s; // source
    double w2  = (double)Wd / 2.0 - 0.5;
    double h2  = (double)Hd / 2.0 - 0.5;
    double ws2 = (double)Ws / 2.0 - 0.5;
    double hs2 = (double)Hs / 2.0 - 0.5;

    for (int y = 0; y < Hd; y++)
    {
        // y-coordinate in dest image relative to center
        y_d = (double)y - h2;
        for (int x = 0; x < Wd; x++)
        {
            x_d = (double)x - w2;
            // Convert fisheye coordinate to cartesian coordinate (equirectangular)
            fish2Eqt(x_d, y_d, &x_s, &y_s, W_rad);
            // Convert source cartesian coordinate to screen coordinate
            x_s += ws2;
            y_s += hs2;
            // Create map
            map_x.at<float>(y, x) = float(x_s);
            map_y.at<float>(y, x) = float(y_s);
        }
    }

} /* fish_2D_map() */


/***************************************************************************************************
*
* Mask creation for cropping image data inside the FOVD circle
*
***************************************************************************************************/
void 
fish_mask_erect( int Ws, int Hs, float in_fovd, cv::Mat &cir_mask )
{
    int wShift = int((Ws * (MAX_FOVD - in_fovd) / MAX_FOVD) / 2.0f);

    // Create Circular mask to crop the input W.R.T. FOVD
    int r = int(float(Ws) / 2.0f - wShift);
    circle(cir_mask, Point(int(Hs / 2), int(Ws / 2)), r, Scalar(255, 255, 255), -1, 8, 0); // fill circle with 0xFF

} /* fish_mask_erect() */


/***************************************************************************************************
*
* Rigid Moving Least Squares Interpolation
*
***************************************************************************************************/
void 
fish_rmls_deform( const cv::Mat &map_x, const cv::Mat &map_y, const cv::Mat &src, cv::Mat &dst )
{
    remap(src, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
} /* fish_rmls_interp2() */


/***************************************************************************************************
*
* Fisheye Light Fall-off Compensation: Scale_Map Construction
*
***************************************************************************************************/
void 
fish_compen_map( const int H, const int W, const cv::Mat &R_pf, cv::Mat &scale_map )
{
    int W_ = floor(W / 2);
    int H_ = floor(H / 2);

    // Create IV quadrant map
    Mat scale_map_quad_4 = Mat::zeros(H_, W_, R_pf.type());
    float da = R_pf.at<float>(0, W_ - 1);
    int x, y;
    float r, a, b;

    for (x = 0; x < W_; x++)
    {
        for (y = 0; y < H_; y++)
        {
            r = floor(sqrt(std::pow(x, 2) + std::pow(y, 2)));
            if (r >= (W_ - 1))
            {
                scale_map_quad_4.at<float>(y, x) = da;
            }
            else
            {
                a = R_pf.at<float>(0, r);
                if ((x < W_) && (y < H_)) // within boundaries
                    b = R_pf.at<float>(0, r + 1);
                else // on boundaries
                    b = R_pf.at<float>(0, r);
                scale_map_quad_4.at<float>(y, x) = (a + b) / 2.0f;
            }
        } // x()
    } // y()

    // Assume Optical Symmetry & Flip
    Mat scale_map_quad_1(scale_map_quad_4.size(), scale_map_quad_4.type());
    Mat scale_map_quad_2(scale_map_quad_4.size(), scale_map_quad_4.type());
    Mat scale_map_quad_3(scale_map_quad_4.size(), scale_map_quad_4.type());
    // 
    cv::flip(scale_map_quad_4, scale_map_quad_1, 0); // quad I, up-down or around x-axis
    cv::flip(scale_map_quad_4, scale_map_quad_3, 1); // quad III, left-right or around y-axis
    cv::flip(scale_map_quad_1, scale_map_quad_2, 1); // quad II, up-down or around x-axis
    // 
    Mat quad_21, quad_34;
    cv::hconcat(scale_map_quad_2, scale_map_quad_1, quad_21);
    cv::hconcat(scale_map_quad_3, scale_map_quad_4, quad_34);
    // 
    cv::vconcat(quad_21, quad_34, scale_map);

} /* fisheye_compen_map() */


/***************************************************************************************************
*
* Fisheye Light Fall-off Compensation
*
***************************************************************************************************/
void 
fish_lighFO_compen( cv::Mat out_img, cv::Mat &in_img, const cv::Mat &scale_map )
{
    Mat rgb_ch[3];
    Mat rgb_ch_double[3];
    Mat out_img_double(in_img.size(), in_img.type());
    cv::split(in_img, rgb_ch);
    rgb_ch[0].convertTo(rgb_ch_double[0], scale_map.type());
    rgb_ch[1].convertTo(rgb_ch_double[1], scale_map.type());
    rgb_ch[2].convertTo(rgb_ch_double[2], scale_map.type());
    //
    rgb_ch_double[0] = rgb_ch_double[0].mul(scale_map); // element-wise multiplication
    rgb_ch_double[1] = rgb_ch_double[1].mul(scale_map);
    rgb_ch_double[2] = rgb_ch_double[2].mul(scale_map);
    cv::merge(rgb_ch_double, 3, out_img_double);

    out_img_double.convertTo(out_img, CV_8U);

} /* fish_lighFO_compen() */


/***************************************************************************************************
*
* Common Preparation
*    Output:   map_x, map_y(unwarping)
*              cir_mask(circular crop)
*
***************************************************************************************************/
void 
fish_common_init( cv::Mat &map_x, cv::Mat &map_y, cv::Mat &cir_mask, cv::Mat &scale_map, cv::Mat &mls_map_x, 
                  cv::Mat &mls_map_y, int Hs, const int Ws, const int Hd, const int Wd, const float fovd )
{
    // Unwarp map_x, map_y, cir_mask
    fish_2D_map(map_x, map_y, Hs, Ws, Hd, Wd, fovd);

    // Create Circular mask to Crop the input W.R.T FOVD (mask all data outside the FOVD circle)
    fish_mask_erect(Ws, Hs, fovd, cir_mask);

    // Scale_Map Reconstruction for Fisheye Light Fall-off Compensation
    int H = Hs;
    int W = Ws;
    int W_ = W / 2;
    Mat x_coor = Mat::zeros(1, W_, CV_32F);
    Mat temp(x_coor.size(), x_coor.type());

    for (int i = 0; i < W_; i++)
    {
        x_coor.at<float>(0, i) = i;
    }

    //  R_pf = P1_ * (x_coor.^5.0) + P2_ * (x_coor.^4.0) + P3_ * (x_coor.^3.0) + 
    //         P4_ * (x_coor.^2.0) + P5_ * x_coor + P6_;
    Mat R_pf = Mat::zeros(x_coor.size(), x_coor.type());
    cv::pow(x_coor, 5.0, temp);
    R_pf = R_pf + P1_ * temp;
    cv::pow(x_coor, 4.0, temp);
    R_pf = R_pf + P2_ * temp;
    cv::pow(x_coor, 3.0, temp);
    R_pf = R_pf + P3_ * temp;
    cv::pow(x_coor, 2.0, temp);
    R_pf = R_pf + P4_ * temp;
    R_pf = R_pf + P5_ * x_coor + P6_;
    //
    // PF_LUT
    cv::divide(1, R_pf, R_pf); //element-wise inverse

    // Generate Scale Map
    fish_compen_map(H, W, R_pf, scale_map);

    // Rigid MLS Interp2 grids
    //3840x1920 resolution (C200 video)
    cv::FileStorage fs("/home/neousys/duweixin/FisheyePro/fisheyestitcher_dwx/build/grid_xd_yd_3840x1920.yml.gz", FileStorage::READ);
    fs["Xd"] >> mls_map_x;
    fs["Yd"] >> mls_map_y;
    fs.release();

} /* fish_common_init() */


/***************************************************************************************************
*
* Adaptive Alignment: Norm XCorr
*    Input: Ref and Template images
*    Output: matchLoc (matching location)
*
***************************************************************************************************/
void 
fish_norm_xcorr( cv::Point2f &matchLoc, const cv::Mat &Ref, cv::Mat &Tmpl, const char* img_window, 
                bool disableDisplay )
{
    double tickStart, tickEnd, runTime;
    Mat img = Ref;
    Mat templ = Tmpl;
    Mat img_display, result;
    img.copyTo(img_display);
    int result_cols = img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;
    result.create(result_rows, result_cols, CV_32FC1);

    // Select Normalized Cross-Correlation as Template Matching Method
    int match_method = CV_TM_CCORR_NORMED;

    // Match template
    matchTemplate(img, templ, result, match_method);
    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

    // Check for peak cross-correlation
    double minVal; double maxVal; Point minLoc; Point maxLoc;

    //Point matchLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
    {
        matchLoc = minLoc;
    }
    else // CV_TM_CCORR_NORMED
    {
        matchLoc = maxLoc;
    }

    if (!disableDisplay)
    {
        rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), 
                  Scalar(0, 255, 0), 2, 8, 0);
        Mat RefTemplCat;
        cv::hconcat(img_display, Tmpl, RefTemplCat);
        imshow(img_window, RefTemplCat);
    }

} /* fish_norm_xcorr() */


/***************************************************************************************************
*
* Construct Control Points for Affine2D estimation
*    Input: Matched Points in the template & ref
*    Output: movingPoints & fixedPoints
*
***************************************************************************************************/
void 
fish_ctrl_points_construct( vector<Point2f> &movingPoints, vector<Point2f> &fixedPoints, 
            const cv::Point2f &matchLocLeft, const cv::Point2f &matchLocRight, const int row_start, 
            const int row_end, const int p_wid, const int p_x1, const int p_x2, const int p_x2_ref )
{
    float x1 = matchLocLeft.x;
    float y1 = matchLocLeft.y;
    float x2 = matchLocRight.x;
    float y2 = matchLocRight.y;

    //-------------------------------------------------------------------------------
    // Construct MovingPoints pRef (matched points of template on reference)
    //-------------------------------------------------------------------------------
    // Left Boundary
    movingPoints.push_back(cv::Point2f(x1, y1 + row_start));                    // pRef_11
    movingPoints.push_back(cv::Point2f(x1 + p_wid, y1 + row_start));            // PRef_12
    movingPoints.push_back(cv::Point2f(x1, y1 + row_end));                      // pRef_13
    movingPoints.push_back(cv::Point2f(x1 + p_wid, y1 + row_end));              // pRef_14
    // Right Boundary
    movingPoints.push_back(cv::Point2f(x2 + p_x2_ref, y2 + row_start));         // pRef_21
    movingPoints.push_back(cv::Point2f(x2 + p_x2_ref + p_wid, y2 + row_start)); // pRef_22
    movingPoints.push_back(cv::Point2f(x2 + p_x2_ref, y2 + row_end));           // pRef_23
    movingPoints.push_back(cv::Point2f(x2 + p_x2_ref + p_wid, y2 + row_end));   // pRef_24

    //-------------------------------------------------------------------------------
    // Construct fixedPoint pTmpl (matched points of template on template)
    //-------------------------------------------------------------------------------
    // Left Boundary
    fixedPoints.push_back(cv::Point2f(p_x1, row_start));          // pTmpl_11
    fixedPoints.push_back(cv::Point2f(p_x1 + p_wid, row_start));  // pTmpl_12
    fixedPoints.push_back(cv::Point2f(p_x1, row_end));            // pTmpl_13
    fixedPoints.push_back(cv::Point2f(p_x1 + p_wid, row_end));    // pTmpl_14
    // Right boundary
    fixedPoints.push_back(cv::Point2f(p_x2, row_start));          // pTmpl_21
    fixedPoints.push_back(cv::Point2f(p_x2 + p_wid, row_start));  // pTmpl_22
    fixedPoints.push_back(cv::Point2f(p_x2, row_end));            // pTmpl_23
    fixedPoints.push_back(cv::Point2f(p_x2 + p_wid, row_end));    // pTmpl_24

} /* fish_ref_point_construct() */


/***************************************************************************************************
*
* Ramp Blending on the right patch
*
***************************************************************************************************/
void 
fish_blend_right( cv::Mat &bg, cv::Mat &bg1, cv::Mat &bg2 )
{
    int h = bg1.size().height;
    int w = bg1.size().width;
    Mat bg_ = Mat::zeros(bg1.size(), CV_32F);
    double alpha1, alpha2;
    Mat bg1_, bg2_;
    bg1.convertTo(bg1_, CV_32F);
    bg2.convertTo(bg2_, CV_32F);
    //
    Mat bgr_bg[3], bgr_bg1[3], bgr_bg2[3];   //destination array
    split(bg1_, bgr_bg1); //split source  
    split(bg2_, bgr_bg2); //split source  
    // 
    bgr_bg[0] = Mat::zeros(bgr_bg1[1].size(), CV_32F);
    bgr_bg[1] = Mat::zeros(bgr_bg1[1].size(), CV_32F);
    bgr_bg[2] = Mat::zeros(bgr_bg1[1].size(), CV_32F);

    for (int r = 0; r < h; r++)
    {
        for (int c = 0; c < w; c++)
        {
            alpha1 = double(c) / double(w);
            alpha2 = 1 - alpha1;
            bgr_bg[0].at<float>(r, c) = alpha1*bgr_bg1[0].at<float>(r, c) + alpha2*bgr_bg2[0].at<float>(r, c);
            bgr_bg[1].at<float>(r, c) = alpha1*bgr_bg1[1].at<float>(r, c) + alpha2*bgr_bg2[1].at<float>(r, c);
            bgr_bg[2].at<float>(r, c) = alpha1*bgr_bg1[2].at<float>(r, c) + alpha2*bgr_bg2[2].at<float>(r, c);
        }
    }
    cv::merge(bgr_bg, 3, bg);
    bg.convertTo(bg, CV_8U);

} /* fish_blend_right() */


/***************************************************************************************************
*
* Ramp Blending on the left patch
*
***************************************************************************************************/
void 
fish_blend_left( cv::Mat &bg, cv::Mat &bg1, cv::Mat &bg2 )
{
    int h = bg1.size().height;
    int w = bg1.size().width;
    Mat bg_ = Mat::zeros(bg1.size(), CV_32F);
    double alpha1, alpha2;
    Mat bg1_, bg2_;
    bg1.convertTo(bg1_, CV_32F);
    bg2.convertTo(bg2_, CV_32F);
    // 
    Mat bgr_bg[3], bgr_bg1[3], bgr_bg2[3];   //destination array
    split(bg1_, bgr_bg1); //split source  
    split(bg2_, bgr_bg2); //split source  
    // 
    bgr_bg[0] = Mat::zeros(bgr_bg1[1].size(), CV_32F);
    bgr_bg[1] = Mat::zeros(bgr_bg1[1].size(), CV_32F);
    bgr_bg[2] = Mat::zeros(bgr_bg1[1].size(), CV_32F);

    for (int r = 0; r < h; r++)
    {
        for (int c = 0; c < w; c++)
        {
            alpha1 = (double(w) - c + 1) / double(w);
            alpha2 = 1 - alpha1;
            bgr_bg[0].at<float>(r, c) = alpha1*bgr_bg1[0].at<float>(r, c) + alpha2*bgr_bg2[0].at<float>(r, c);
            bgr_bg[1].at<float>(r, c) = alpha1*bgr_bg1[1].at<float>(r, c) + alpha2*bgr_bg2[1].at<float>(r, c);
            bgr_bg[2].at<float>(r, c) = alpha1*bgr_bg1[2].at<float>(r, c) + alpha2*bgr_bg2[2].at<float>(r, c);
        }
    }
    cv::merge(bgr_bg, 3, bg);
    bg.convertTo(bg, CV_8U);

} /* fish_blend_left() */


/***************************************************************************************************
*
* Blending aligned images
*    Input: aligned images
*    Output: blended pano
*
***************************************************************************************************/
void 
fish_blend( cv::Mat &left_img, cv::Mat &right_img_aligned, cv::Mat &pano )
{
    // Read YML
    Mat post;
    cv::FileStorage fs("/home/neousys/duweixin/FisheyePro/fisheyestitcher_dwx/build/post_find.yml", FileStorage::READ);
    fs["post_ret"] >> post; // 1772 x 1
    fs.release();

    // Mask
    Mat mask = imread("/home/neousys/duweixin/FisheyePro/fisheyestitcher_dwx/build/mask_1920x1920_fovd_187.jpg", IMREAD_COLOR);
    int H = mask.size().height;
    int W = mask.size().width;

    int Worg = 1920;
    int imH = left_img.size().height;
    int imW = left_img.size().width;
    Mat left_img_cr = left_img(Rect(imW / 2 + 1 - Worg / 2, 0, Worg, imH));
    int  sideW = 45;
    Mat left_blend, right_blend;

    for (int r = 0; r < H; r++)
    {
        int p = post.at<float>(r, 0);
        if (p == 0)
        {
            continue;
        }
        // Left boundary
        Mat lf_win_1 = left_img_cr(Rect(p - sideW, r, 2 * sideW, 1));
        Mat rt_win_1 = right_img_aligned(Rect(p - sideW, r, 2 * sideW, 1));
        // Right boundary
        Mat lf_win_2 = left_img_cr(Rect((W - p - sideW), r, 2 * sideW, 1));
        Mat rt_win_2 = right_img_aligned(Rect((W - p - sideW), r, 2 * sideW, 1));
        // Blend(ramp)
        Mat bleft, bright;
        fish_blend_left(bleft, lf_win_1, rt_win_1);
        fish_blend_right(bright, lf_win_2, rt_win_2);
        // Update left boundary
        bleft.copyTo(lf_win_1);
        bleft.copyTo(rt_win_1);
        // Update right boundary
        bright.copyTo(lf_win_2);
        bright.copyTo(rt_win_2);
    }

    // BLENDING
    Mat mask_ = mask(Rect(0, 0, mask.size().width, mask.size().height - 2));
    Mat mask_n;
    bitwise_not(mask_, mask_n);
    bitwise_and(left_img_cr, mask_, left_img_cr); // Left image
    // 
    Mat temp1 = left_img(Rect(0, 0, (imW / 2 - Worg / 2), imH));
    Mat temp2 = left_img(Rect((imW / 2 + Worg / 2), 0, (imW / 2 - Worg / 2), imH));
    Mat t;
    cv::hconcat(temp1, left_img_cr, t);
    cv::hconcat(t, temp2, left_img);
    // 
    bitwise_and(right_img_aligned, mask_n, right_img_aligned); // Right image
    //
    pano = left_img;
    Mat temp = pano(Rect((imW / 2 - Worg / 2), 0, Worg, imH));
    Mat t2;
    bitwise_or(temp, right_img_aligned, t2);
    t2.copyTo(temp); // updated pano

} /* fish_blend() */


/***************************************************************************************************
*
* Single frame stitch
*
****************************************************************************************************/
void 
fish_stitch_one( cv::Mat &pano, cv::Mat &in_img_L, cv::Mat &in_img_R, const cv::Mat &cir_mask, 
        cv::Mat &scale_map, const cv::Mat &map_x, const cv::Mat &map_y, const cv::Mat &mls_map_x, 
        const cv::Mat &mls_map_y, const int Hs, const int Ws, const int Hd, const int Wd, 
        const int W_in, const bool disable_light_compen, const bool disable_refine_align )
{
    Mat left_unwarped, right_unwarped;

    //--------------------------------------------------------------------------
    // Circular Crop
    //--------------------------------------------------------------------------
    bitwise_and(in_img_L, cir_mask, in_img_L); // Left image
    bitwise_and(in_img_R, cir_mask, in_img_R); // Right image

    //--------------------------------------------------------------------------
    // Light Fall-off Compensation
    //--------------------------------------------------------------------------
    Mat left_img_compensated(in_img_L.size(), in_img_L.type());
    Mat right_img_compensated(in_img_R.size(), in_img_R.type());
    if (disable_light_compen)
    {
        left_img_compensated = in_img_L;
        right_img_compensated = in_img_R;
    }
    else
    {
        fish_lighFO_compen(left_img_compensated, in_img_L, scale_map);
        fish_lighFO_compen(right_img_compensated, in_img_R, scale_map);
    }

    //--------------------------------------------------------------------------
    // Fisheye Unwarping
    //--------------------------------------------------------------------------
    fish_unwarp(map_x, map_y, left_img_compensated, left_unwarped);
    fish_unwarp(map_x, map_y, right_img_compensated, right_unwarped);

    //--------------------------------------------------------------------------
    // Rigid Moving Least Squares Deformation
    //--------------------------------------------------------------------------
    Mat rightImg_crop, rightImg_mls_deformed;
    rightImg_crop = right_unwarped(Rect(int(Wd / 2) - (W_in / 2), 0, W_in, Hd - 2)); // notice on (Hd-2) --> become: (Hd)
    fish_rmls_deform(mls_map_x, mls_map_y, rightImg_crop, rightImg_mls_deformed);


    //--------------------------------------------------------------------------
    // Rearrange Image for Adaptive Alignment
    //--------------------------------------------------------------------------
    Mat temp1 = left_unwarped(Rect(0, 0, int(Wd / 2), Hd - 2));
    Mat temp2 = left_unwarped(Rect(int(Wd / 2), 0, int(Wd / 2), Hd - 2));
    Mat left_unwarped_arr; // re-arranged left unwarped
    cv::hconcat(temp2, temp1, left_unwarped_arr);
    Mat leftImg_crop;
    leftImg_crop = left_unwarped_arr(Rect(int(Wd / 2) - (W_in / 2), 0, W_in, Hd - 2)); 
    uint16_t crop = uint16_t(0.5 * Ws * (195.0 - 180.0) / 195.0); // half overlap region

    //--------------------------------------------------------------------------
    // PARAMETERS (hard-coded) for C200 videos 
    //--------------------------------------------------------------------------
    uint16_t p_wid = 55;
    uint16_t p_x1 = 90 - 15;
    uint16_t p_x2 = 1780 - 5; 
    uint16_t p_x1_ref = 2 * crop;
    uint16_t row_start = 590;
    uint16_t row_end = 1320;
    uint16_t p_x2_ref = Ws - 2 * crop + 1;
    // 
    Mat Ref_1, Ref_2, Tmpl_1, Tmpl_2;
    Ref_1  = leftImg_crop(Rect(0, row_start, p_x1_ref, row_end - row_start));
    Ref_2  = leftImg_crop(Rect(p_x2_ref, row_start, Ws - p_x2_ref, row_end - row_start));
    Tmpl_1 = rightImg_mls_deformed(Rect(p_x1, row_start, p_wid, row_end - row_start));
    Tmpl_2 = rightImg_mls_deformed(Rect(p_x2, row_start, p_wid, row_end - row_start));


    //--------------------------------------------------------------------------
    // Adaptive Alignment (Norm XCorr)
    //--------------------------------------------------------------------------
    bool disableDisplay = 1;   // 1: display off, 0: display on
    const char* name_bound_1 = "Matching On Left Boundary";
    const char* name_bound_2 = "Matching On Right Boundary";


    cv::Mat warpedRightImg;
    if (disable_refine_align)
    {
        warpedRightImg = rightImg_mls_deformed;
    }
    else
    {
        //--------------------------------------------------------------
        // Normalized XCorr
        //--------------------------------------------------------------
        cv::Point2f matchLocLeft, matchLocRight;
        fish_norm_xcorr(matchLocLeft, Ref_1, Tmpl_1, name_bound_1, disableDisplay); // Left boundary
        fish_norm_xcorr(matchLocRight, Ref_2, Tmpl_2, name_bound_2, disableDisplay); // Right boundary


        vector<Point2f> movingPoints;  // matched points in Refs  
        vector<Point2f> fixedPoints;   // matched points in Templates

        //--------------------------------------------------------------
        // Construct control points for estimation
        //--------------------------------------------------------------
        fish_ctrl_points_construct(movingPoints, fixedPoints, matchLocLeft, matchLocRight, row_start, row_end, p_wid, p_x1, p_x2, p_x2_ref);

        cv::Mat tform_refine_mat;
        //--------------------------------------------------------------
        // Estimate affine matrix
        //--------------------------------------------------------------
        tform_refine_mat = cv::findHomography(fixedPoints, movingPoints, 0);

        //--------------------------------------------------------------
        // Warp Image
        //--------------------------------------------------------------
        cv::warpPerspective(rightImg_mls_deformed, warpedRightImg, tform_refine_mat, 
                            rightImg_mls_deformed.size(), INTER_LINEAR);


    } // Normalized xcorr

    //--------------------------------------------------------------
    // Blend Images
    //--------------------------------------------------------------
    fish_blend(left_unwarped_arr, warpedRightImg, pano);


} /* fish_stitch_one() */

/***************************************************************************************************
*
* 获得方形图
*
***************************************************************************************************/


Mat GetSquareImage1( const cv::Mat& img, int target_width = 1024 )
{
    int width = img.cols,
       height = img.rows;

    Mat square = cv::Mat::zeros( target_width, target_width, img.type() );

    int max_dim = ( width >= height ) ? width : height;
    float scale = ( ( float ) target_width ) / max_dim;
    Rect roi;
    if ( width >= height )
    {
        roi.width = target_width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = (( target_width - roi.height ) / 2) - 20;
    }
    else
    {
        roi.y = 0;
        roi.height = target_width;
        roi.width = width * scale;
        roi.x = ( target_width - roi.width ) / 2;
    }

    cv::resize( img, square( roi ), roi.size() );

    return square;
}
Mat GetSquareImage2( const cv::Mat& img, int target_width = 1024 )
{
    int width = img.cols,
       height = img.rows;

    Mat square = cv::Mat::zeros( target_width, target_width, img.type() );

    int max_dim = ( width >= height ) ? width : height;
    float scale = ( ( float ) target_width ) / max_dim;
    Rect roi;
    if ( width >= height )
    {
        roi.width = target_width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = (( target_width - roi.height ) / 2) + 20 ;
    }
    else
    {
        roi.y = 0;
        roi.height = target_width;
        roi.width = width * scale;
        roi.x = ( target_width - roi.width ) / 2;
    }

    cv::resize( img, square( roi ), roi.size() );

    return square;
}

/***************************************************************************************************
*
* Main()
*
***************************************************************************************************/


int main(int argc,char* argv[])
{
/***************************************************************************************************
*
* 一个窗口显示两个摄像头
*
***************************************************************************************************/
    Mat frame1, frame2,ROIImage1, ROIImage2,mask1,mask2,gray_Image1, gray_Image2,  CutFrame1,  CutFrame2;
	
    float fovd = 195.0f;
    bool  disable_light_compen = 1;
    bool  disable_refine_align = 1;

    int W_in = 1920; // default: video 3840X1920
    
    Mat in_img, in_img_L, in_img_R;
    
    VideoCapture capture1(0);
	VideoCapture capture2(1);
	capture1.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	capture1.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	capture2.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	capture2.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
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
	Mat frame3(1024,2048,CV_8UC3,Scalar(0,0,0));
    Mat temp_frame3(1920, 3840, frame3.type());

    //**************************************************************************
    // PreComputation
    //**************************************************************************
    in_img = temp_frame3;
    int Worg = in_img.size().width;
    int Horg = in_img.size().height;
    // Source image
    int Ws = int(Worg / 2);
    int Hs = Horg;
    // Destination pano
    int Wd = int(Ws * 360.0 / MAX_FOVD);
    int Hd = int(Wd / 2);
    //--------------------------------------------------------------------------
    // Initialization of all precomputed data
    //--------------------------------------------------------------------------
    Mat map_x(Hd, Wd, CV_32FC1);
    Mat map_y(Hd, Wd, CV_32FC1);
    Mat cir_mask(Hs, Ws, in_img.type());
    Mat scale_map(Hs, Ws, CV_32F);
    Mat mls_map_x, mls_map_y;
    //
    fish_common_init(map_x, map_y, cir_mask, scale_map, mls_map_x, mls_map_y, Hs, Ws, Hd, Wd, 195);
    //
    in_img.release();
    Mat pano;

    while (true)
	{
		capture1 >> frame1;
		capture2 >> frame2;

        CutFrame1 = frame1(Rect(118,0,frame1.cols - 256, frame1.rows));
        CutFrame2 = frame2(Rect(106,0,frame2.cols - 256, frame2.rows));

        Mat res_CutFrame1(720, 1024, CutFrame1.type());
        resize(CutFrame1, res_CutFrame1, res_CutFrame1.size(), INTER_LINEAR);
        Mat res_CutFrame2(720, 1024, CutFrame2.type());
        resize(CutFrame2, res_CutFrame2, res_CutFrame2.size(), INTER_LINEAR);

        Mat rect_frame1 = GetSquareImage1(res_CutFrame1);
        Mat rect_frame2 = GetSquareImage2(res_CutFrame2);

        ROIImage1 = frame3(Rect(0,0,1024,1024));
		ROIImage2 = frame3(Rect(1024, 0, 1024, 1024));
		// imshow("【ROIImage1】", ROIImage1);
		// imshow("【ROIImage2】", ROIImage2);
		cvtColor(rect_frame1, gray_Image1,CV_RGB2GRAY);
		cvtColor(rect_frame2, gray_Image2, CV_RGB2GRAY);
		mask1 = gray_Image1;
		mask2 = gray_Image2;
		rect_frame1.copyTo(ROIImage1, mask1);
		rect_frame2.copyTo(ROIImage2, mask2);

        // imshow("【视频窗口一】", res_CutFrame1);
        // imshow("【视频窗口二】", res_CutFrame2);
        // resize frame3
        
        Mat res_frame3(1920, 3840, frame3.type());
        resize(frame3, res_frame3, res_frame3.size(), INTER_LINEAR);

        // //--------------------------------------------------------------------------
        // // Read frames and process
        // //--------------------------------------------------------------------------
        //     // Read input fisheye images
        // in_img = res_frame3;
        // in_img_L = in_img(Rect(0, 0, Worg / 2, Horg));        // left fisheye
        // in_img_R = in_img(Rect(Worg / 2, 0, Worg / 2, Horg)); // right fisheye
        //     // Stitch
        // fish_stitch_one(pano, in_img_L, in_img_R, cir_mask, scale_map, map_x, map_y, 
        //     mls_map_x, mls_map_y, Hs, Ws, Hd, Wd, W_in, disable_light_compen, disable_refine_align);

        // Mat resize_pano(800,1600, pano.type());
        // resize(pano, resize_pano, resize_pano.size(), 0, 0, INTER_LINEAR);

        // // imwrite("/home/neousys/duweixin/FisheyePro/fisheyestitcher_dwx/output/*.jpg", frame3);

        // imshow("【总窗口】", resize_pano);
        // resize_pano.release();
        // waitKey(1);

        // tiaoshi :
        imshow("【frame1】", frame3);
        // imshow("【frame2】", rect_frame2);
        waitKey(2);
        imwrite("/home/neousys/duweixin/FisheyePro/fisheyestitcher_dwx/output/frame3.jpg", frame3);
        // imwrite("/home/neousys/duweixin/FisheyePro/fisheyestitcher_dwx/output/rect_frame2.jpg", rect_frame2);
        waitKey(1);

    }
    return 0;

}
