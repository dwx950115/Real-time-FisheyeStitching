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

#define     MY_DEBUG      0
#define     MAX_FOVD      210.0f

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
* Mask creation for cropping image data inside the FOVD circle
*
***************************************************************************************************/
void 
fish_mask_erect( const int Ws, const int Hs, const float in_fovd, 
                 cv::Mat &cir_mask, cv::Mat &inner_cir_mask )
{
    Mat cir_mask_ = Mat(Hs, Ws, CV_8UC3);
    Mat inner_cir_mask_ = Mat(Hs, Ws, CV_8UC3);

    int wShift = static_cast<int>(floor(((Ws * (MAX_FOVD - in_fovd) / MAX_FOVD) / 2.0f)));

    // Create Circular mask to crop the input W.R.T. FOVD
    int r1 = static_cast<int>(floor((double(Ws) / 2.0)));
    int r2 = static_cast<int>(floor((double(Ws) / 2.0 - wShift * 2.0)));
    circle(cir_mask_,       Point(int(Hs / 2), int(Ws / 2)), r1, Scalar(255, 255, 255), -1, 8, 0); // fill circle with 0xFF
    circle(inner_cir_mask_, Point(int(Hs / 2), int(Ws / 2)), r2, Scalar(255, 255, 255), -1, 8, 0); // fill circle with 0xFF

    cir_mask_.convertTo(cir_mask, CV_8UC3);
    inner_cir_mask_.convertTo(inner_cir_mask, CV_8UC3);
} /* fish_mask_erect() */

/***************************************************************************************************
*
* create blend mask for blend post
*
***************************************************************************************************/
void
fish_create_blend_mask( const cv::Mat &cir_mask, const cv::Mat &inner_cir_mask,
                  const int Ws, const int Hs, const int Wd, const int Hd,
                  const Mat &map_x, const Mat &map_y,
                  cv::Mat &binary_mask, std::vector<int> &blend_post )
{
    Mat inner_cir_mask_n;
    Mat ring_mask, ring_mask_unwarped;

    int Ws2 = static_cast<int>(Ws / 2);
    int Hs2 = static_cast<int>(Hs / 2);
    int Wd2 = static_cast<int>(Wd / 2);
    bitwise_not(inner_cir_mask, inner_cir_mask_n);

    cir_mask.copyTo(ring_mask, inner_cir_mask_n);

#if MY_DEBUG
    imwrite("./output/ring_mask.jpg", ring_mask);
#endif
    
    remap(ring_mask, ring_mask_unwarped, map_x, map_y, INTER_LINEAR, 
          BORDER_CONSTANT, Scalar(0, 0, 0));
    
    Mat mask_ = ring_mask_unwarped(Rect(Wd2-Ws2, 0, Ws, Hd)); 
    mask_.convertTo(mask_, CV_8UC3);

    int H_ = mask_.size().height;
    int W_ = mask_.size().width;

    int ridx, cidx;

    const int first_zero_col = 120; // first cidx that mask value is zero
    const int first_zero_row =  45; // first ridx that mask value is zero

    // Clean up
    for( ridx=0; ridx < H_; ++ridx)
    {
        if( cidx < first_zero_col || cidx > W_-first_zero_col )
        {
            mask_.at<Vec3b>(Point(cidx,ridx)) = Vec3b(255,255,255);
        }
    }
    for( ridx=0; ridx < H_; ++ridx )
    {
        for( cidx=0; cidx < W_; ++cidx )
        {
            if( (ridx < (static_cast<int>(H_/2)) ) &&
                (cidx > first_zero_col-1) && 
                (cidx < W_-first_zero_col+1) )
            { 
                mask_.at<Vec3b>(Point(cidx,ridx)) = Vec3b(0, 0, 0);
            }
        }
    }

    // Create blend_post
    int offset = 15;
    for( ridx=0; ridx < H_; ++ridx )
    {
        if( ridx > H_-first_zero_row )
        {
            blend_post.push_back( 0 ); 
            continue;
        }
        for( cidx=first_zero_col-10; cidx < W_/2+10; ++cidx )
        {
            Vec3b color = mask_.at<Vec3b>(Point(cidx,ridx));
            if( color == Vec3b(0,0,0))
            {
                blend_post.push_back( cidx - offset );
                break; 
            }
        } 
    }

    mask_.convertTo( binary_mask, CV_8UC3 );

#if MY_DEBUG
    cout << "size mask_ = " << mask_.size() << ", type = " << mask_.type() << ", ch = " << mask_.channels() << "\n";
    imwrite("./output/mask.jpg", mask_);
#endif
}


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
fish_blend( const cv::Mat &left_img, const cv::Mat &right_img_aligned, cv::Mat &pano,
            const cv::Mat &binary_mask, const std::vector<int> &blend_post )
{ 
    Mat mask = binary_mask;

    int H = mask.size().height;
    int W = mask.size().width;

    int Worg = 1920;
    int imH = left_img.size().height;
    int imW = left_img.size().width;
    Mat left_img_cr = left_img(Rect(imW / 2 + 1 - Worg / 2, 0, Worg, imH));
    int  sideW = 45; //45
    // Mat left_blend, right_blend;

#if MY_DEBUG
    imwrite("./output/left_img.jpg", left_img);
    imwrite("./output/left_img_cr.jpg", left_img_cr);
#endif

    for (int r = 0; r < H; ++r)
    {
        int p = blend_post[r];

        if (p == 0)
        {
            continue;
        }
        // Left boundary
        Mat lf_win_1 = left_img_cr(Rect(p - sideW, r, 2 * sideW, 1)); // kuan du 90
        Mat rt_win_1 = right_img_aligned(Rect(p - sideW, r, 2 * sideW, 1));  //kuandu 90
        // Right boundary
        Mat lf_win_2 = left_img_cr(Rect((W - p - sideW), r, 2 * sideW, 1)); //kuandu 90
        Mat rt_win_2 = right_img_aligned(Rect((W - p - sideW), r, 2 * sideW, 1)); // kuandu 90
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

#if MY_DEBUG
    imwrite("./output/left_crop_blend.jpg", left_img_cr);
    imwrite("./output/right_blend.jpg", right_img_aligned);
#endif

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
#if MY_DEBUG
    imwrite("./output/pano_update.jpg", pano);
#endif
} /* fish_blend() */


int main( int argc, char** argv )
{	
/***************************************************************************************************
*
* 一个窗口显示两个摄像头
*
***************************************************************************************************/
    Mat frame1, frame2,ROIImage1, ROIImage2,mask1,mask2,gray_Image1, gray_Image2,  CutFrame1,  CutFrame2;
	
    float fovd = 210.0f;
    bool  disable_light_compen = 1;
    bool  disable_refine_align = 1;

    int W_in = 1920; // default: video 3840X1920
    
    Mat in_img, in_img_L, in_img_R;
    
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

	Mat frame3(1140,2280,CV_8UC3,Scalar(0,0,0));
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
    Mat scale_map(Hs, Ws, CV_32F);
    // Mat mls_map_x, mls_map_y;

    //--------------------------------------------------------------------------
    // Scale_Map Reconstruction for Fisheye Light Fall-off Compensation
    //--------------------------------------------------------------------------
    Mat x_coor = Mat::zeros(1, Ws/2, CV_32F);
    Mat temp(x_coor.size(), x_coor.type());

    for (int i = 0; i < Ws/2; i++)
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
    // PF_LUT
    cv::divide(1, R_pf, R_pf); //element-wise inverse

    //--------------------------------------------------------------------------
    // Unwarp map_x, map_y, Light Fall-off Compensation
    //--------------------------------------------------------------------------
	fish_2D_map(map_x, map_y, Hs, Ws, Hd, Wd, fovd);
	CV_Assert( (Hs % 2 == 0) && (Ws % 2 == 0) );
    //--------------------------------------------------------------------------
    // Create Circular mask to Crop the input W.R.T FOVD 
    // (mask all data outside the FOVD circle)
    //--------------------------------------------------------------------------
    Mat cir_mask(Hs, Ws, in_img.type());
    Mat inner_cir_mask, binary_mask;
    std::vector<int> blend_post; // blending
    double inner_fovd = 183.0;
    fish_mask_erect(Ws, Hs, inner_fovd, cir_mask, inner_cir_mask);
    fish_create_blend_mask( cir_mask, inner_cir_mask, Ws, Hs, Wd, Hd, 
                      map_x, map_y, binary_mask, blend_post );
    //--------------------------------------------------------------------------
    // Light Fall-off Compensation
    //--------------------------------------------------------------------------
	fish_compen_map(Hs, Ws, R_pf, scale_map);
	in_img.release();
	Mat pano;

    while (true)
	{
		capture1.grab();
		capture2.grab();
		capture1.retrieve(frame1);
		capture2.retrieve(frame2);

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

        // resize frame3
        Mat res_frame3(1920, 3840, frame3.type());
        resize(frame3, res_frame3, res_frame3.size(), INTER_LINEAR);

        //--------------------------------------------------------------------------
        // Read frames and process
        //--------------------------------------------------------------------------
        // Read input fisheye images
        in_img = res_frame3;
        in_img_L = in_img(Rect(0, 0, Worg / 2, Horg));        // left fisheye
        in_img_R = in_img(Rect(Worg / 2, 0, Worg / 2, Horg)); // right fisheye
        // Stitch
        //--------------------------------------------------------------------------
    	// Fisheye Unwarping
   		//--------------------------------------------------------------------------
   		Mat left_unwarped, right_unwarped;
	    Mat left_img_compensated(in_img_L.size(), in_img_L.type());
   		Mat right_img_compensated(in_img_R.size(), in_img_R.type());
        fish_lighFO_compen(left_img_compensated, in_img_L, scale_map);
        fish_lighFO_compen(right_img_compensated, in_img_R, scale_map);
    	fish_unwarp(map_x, map_y, left_img_compensated, left_unwarped);
    	fish_unwarp(map_x, map_y, right_img_compensated, right_unwarped);
	    //--------------------------------------------------------------------------
		// left_unwarped Image for Adaptive Alignment
		//--------------------------------------------------------------------------
		Mat temp1 = left_unwarped(Rect(0, 0, int(Wd / 2), Hd -2 ));
		Mat temp2 = left_unwarped(Rect(int(Wd / 2), 0, int(Wd / 2), Hd -2));
		Mat left_unwarped_arr; // re-arranged left unwarped
		cv::hconcat(temp2, temp1, left_unwarped_arr);
		// Mat leftImg_crop;
		// leftImg_crop = left_unwarped_arr(Rect(int(Wd / 2) - (W_in / 2), 0, W_in, Hd -2)); 
		// uint16_t crop = uint16_t(0.5 * Ws * (210.0 - 180.0) / 210.0); // half overlap region
	    //--------------------------------------------------------------------------
	    // right_unwarped Image for Adaptive Alignment
	    //--------------------------------------------------------------------------
	    Mat rightImg_crop;
	    rightImg_crop = right_unwarped(Rect(int(Wd / 2) - (W_in / 2), 0, W_in, Hd -2)); 
	    //--------------------------------------------------------------------------
	    // PARAMETERS (hard-coded) for dual-fisheye
	    //--------------------------------------------------------------------------

	    //--------------------------------------------------------------
	    // Blend Images
	    //--------------------------------------------------------------
	    fish_blend(left_unwarped_arr, rightImg_crop, pano, binary_mask, blend_post);

        Mat resize_pano(800,1600, pano.type());
        resize(pano, resize_pano, resize_pano.size(), 0, 0, INTER_LINEAR);



        // //debug :
		// imshow("【frame1】", frame1);
		// imwrite("/home/neousys/duweixin/FisheyePro/fisheye_test_program/frame1.jpg",frame1);
		// imshow("【frame2】", frame2);
		// imwrite("/home/neousys/duweixin/FisheyePro/fisheye_test_program/frame2.jpg",frame2);
		// imshow("【CutFrame1】", CutFrame1);
		// imshow("【CutFrame2】", CutFrame2);
        // imshow("【视频窗口一】", res_CutFrame1);
        // imshow("【视频窗口二】", res_CutFrame2);
		// imshow("【ROIImage1】", ROIImage1);
		// imshow("【ROIImage2】", ROIImage2);
        // imshow("【frame3】", frame3);
        // imwrite("/home/neousys/duweixin/FisheyePro/fisheye_test_program/frame3.jpg", frame3);
        // imshow("in_img_L",left_unwarped);
        // imwrite("/home/neousys/duweixin/FisheyePro/fisheye_test_program/left_unwarped.jpg", left_unwarped);
        // imshow("in_img_R",right_unwarped);
        // imwrite("/home/neousys/duweixin/FisheyePro/fisheye_test_program/right_unwarped.jpg", right_unwarped);
        imshow("【总窗口】", resize_pano);


        resize_pano.release();
        waitKey(1);

    }
	return 0;
}
