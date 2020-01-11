#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <pthread.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp> // for findHomograph
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;
using namespace Eigen;
using Eigen::MatrixXd;

#define     MY_DEBUG        0
#define     TEST            0
#define     capture_source  1
#define     pi              3.1415926

/***************************************************************************************************
*
* Fisheye Unwarping 鱼眼展开
*   Unwarp source fisheye -> 360x180 equirectangular
*
***************************************************************************************************/
void
fish_angle_regulation(const cv::Mat &map_angle_x, const cv::Mat &map_angle_y, const cv::Mat &src, cv::Mat &dst)
{
    remap(src, dst, map_angle_x, map_angle_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
}

/***************************************************************************************************
*
* Map 2D fisheye image to 2D projected sphere and transform the angle
*
***************************************************************************************************/
void
euler_angle_rotation(float cam_yaw, float cam_pitch, float cam_roll, Matrix3f& R)
{
    Matrix3f R_x(3, 3), R_y(3, 3), R_z(3, 3), R_y_x(3, 3);
    R_x << 1, 0, 0,0, cos(cam_yaw), -sin(cam_yaw),0, sin(cam_yaw), cos(cam_yaw);
    R_y << cos(cam_pitch), 0, sin(cam_pitch),0, 1, 0,-sin(cam_pitch), 0, cos(cam_pitch);
    R_z << cos(cam_roll), -sin(cam_roll), 0,sin(cam_roll),cos(cam_roll), 0,0, 0, 1;
    R_y_x = R_y * R_x;
    R = R_z * R_y_x;
}
void
pixel2point(int u, int v, float cam_fx, float cam_fy, float cam_u0, float cam_v0, MatrixXf& points, int Hs, int Ws)
{
    float pixel_r = sqrt(pow((u - cam_u0), 2)+pow((v - cam_v0), 2));
    float pixel_theta = pixel_r/cam_fx;
    float p_z = cos(pixel_theta);
    float p_x = sin(pixel_theta)*(u - cam_u0)/pixel_r;
    float p_y = sin(pixel_theta)*(v - cam_v0)/pixel_r;

    points((v*Ws)+u,0) = p_x;
    points((v*Ws)+u,1) = p_y;
    points((v*Ws)+u,2) = p_z;
}

void
point2pixel(MatrixXf& points_transd, MatrixXf& map_x_matrix, MatrixXf& map_y_matrix, 
            float cam_fx, float cam_fy, float cam_u0, float cam_v0, int Hs, int Ws,float p1, float p2, float p3, float p4)
{
    int row = 1, col = 1;
    for (int i=0; i<Hs*Ws; i++)
    {
        float point_theta_init = acos(points_transd(i,2)/(sqrt(pow(points_transd(i,0),2)+pow(points_transd(i,1),2)+pow(points_transd(i,2),2))));
        float point_theta = point_theta_init*(1+p1*pow(point_theta_init,2)+p2*pow(point_theta_init,4)+p3*pow(point_theta_init,6)+p4*pow(point_theta_init,8));
        float px, py;
        // if (point_theta > pi)
        // {
        //     point_theta = pi*2-point_theta;
        // }
        if (points_transd(i,0)==0 || points_transd(i,1) == 0)
        {
            px = points_transd(i,0)*cam_fx*point_theta/sqrt(pow(0.001, 2)+pow(0.001, 2))+cam_u0;
            py = points_transd(i,1)*cam_fy*point_theta/sqrt(pow(0.001, 2)+pow(0.001, 2))+cam_v0;
        }
        px = points_transd(i,0)*cam_fx*point_theta/sqrt(pow(points_transd(i,0),2)+pow(points_transd(i,1),2))+cam_u0;
        py = points_transd(i,1)*cam_fy*point_theta/sqrt(pow(points_transd(i,0),2)+pow(points_transd(i,1),2))+cam_v0;

        map_x_matrix(row-1, col-1) = px;
        map_y_matrix(row-1, col-1) = py;
        col++;
        if ((i+1)%Ws == 0)
        {
            row++;
            col = 1;
        }
    }
}

void
lola2point(int u, int v, int Wd, int Hd, MatrixXf& points)
{
    float lati = (2*(u+0.001)/Wd-1)*pi;
    float longti = ((v+0.001)/Hd-0.5)*pi;
    float p_y = sin(longti);
    float p_z = cos(longti)*cos(lati);
    float p_x = cos(longti)*sin(lati);

    points((v*Wd)+u,0) = p_x;
    points((v*Wd)+u,1) = p_y;
    points((v*Wd)+u,2) = p_z;
}

void 
angle_transformation( cv::Mat &map_angle_x, cv::Mat &map_angle_y, float cam_yaw, float cam_pitch, float cam_roll,
                    float cam_fx, float cam_fy, float cam_u0, float cam_v0,
                    int Hs, int Ws,float p1, float p2, float p3, float p4 )
{
    Matrix3f R(3, 3), R_inversed(3, 3);
    MatrixXf points_transd(Hs*Ws, 3), points(Hs*Ws, 3), point(Hs*Ws, 3);
    MatrixXf map_x_matrix(Hs, Ws), map_y_matrix(Hs, Ws);

    euler_angle_rotation(cam_yaw, cam_pitch, cam_roll, R);
    R_inversed = R.transpose();
    // R_inversed = R.inverse();
    for(int v=0; v<Hs; v++)
    {
        for(int u=0; u<Ws; u++)
        {
            pixel2point(u, v, cam_fx, cam_fy, cam_u0, cam_v0, points, Hs, Ws);
        }
    }
    points_transd = (R_inversed * points.transpose()).transpose();
    point2pixel(points_transd, map_x_matrix, map_y_matrix, cam_fx, cam_fy, cam_u0, cam_v0, Hs, Ws, p1,  p2,  p3,  p4);

    eigen2cv(map_x_matrix, map_angle_x);
    eigen2cv(map_y_matrix, map_angle_y);
}

void
angle_transformation_unwarping_cam1(cv::Mat &map_angle_x, cv::Mat &map_angle_y, float cam_yaw, float cam_pitch, float cam_roll,
                                float cam_fx, float cam_fy, float cam_u0, float cam_v0, int Hs, int Ws, int Hd, int Wd,
                                float p1, float p2, float p3, float p4)
{
    MatrixXf points(Hd*Wd, 3);
    MatrixXf map_x_matrix(Hd, Wd), map_y_matrix(Hd, Wd);
    for(int v=0; v<Hd; v++)
    {
        for(int u=0; u<Wd; u++)
        {
            lola2point(u, v, Wd, Hd, points);
        }
    }
    point2pixel(points, map_x_matrix, map_y_matrix, cam_fx, cam_fy, cam_u0, cam_v0, Hd, Wd, p1,  p2,  p3,  p4);

    eigen2cv(map_x_matrix, map_angle_x);    
    eigen2cv(map_y_matrix, map_angle_y);
}
void
angle_transformation_unwarping_cam2(cv::Mat &map_angle_x, cv::Mat &map_angle_y, float cam_yaw, float cam_pitch, float cam_roll,
                                float cam_fx, float cam_fy, float cam_u0, float cam_v0, int Hs, int Ws, int Hd, int Wd,
                                float p1, float p2, float p3, float p4,
                                Matrix3f& R_l2c1, Matrix3f& R_l2c2)
{
    MatrixXf points(Hd*Wd, 3),points_transd(Hd*Wd,3);
    MatrixXf map_x_matrix(Hd, Wd), map_y_matrix(Hd, Wd);
    Matrix3f R_l2c2_inversed,R_cam12cam2,R_all;
    for(int v=0; v<Hd; v++)
    {
        for(int u=0; u<Wd; u++)
        {
            lola2point(u, v, Wd, Hd, points);
        }
    }
    R_l2c2_inversed = R_l2c2.inverse();
    euler_angle_rotation(cam_yaw, cam_pitch, cam_roll, R_cam12cam2);
    R_all = R_cam12cam2*(R_l2c1*R_l2c2_inversed);
    points_transd = (R_all*(points.transpose())).transpose();
    point2pixel(points_transd, map_x_matrix, map_y_matrix, cam_fx, cam_fy, cam_u0, cam_v0, Hd, Wd, p1,  p2,  p3,  p4);

    eigen2cv(map_x_matrix, map_angle_x);    
    eigen2cv(map_y_matrix, map_angle_y);
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
            // alpha1 = 1- (cos(double(c)*pi / double(w))+1)/2;
            // alpha2 = 1 - alpha1;
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
            // alpha1 = 1 - (cos((double(w) - c + 1)*pi / double(w))+1)/2;
            // alpha2 = 1 - alpha1;
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
fish_blend_directly(const cv::Mat &left_unwarped, const cv::Mat &right_unwarped, cv::Mat &pano)
{
    int imH = left_unwarped.size().height;
    int imW = left_unwarped.size().width;
    Mat left_img_crop_left = left_unwarped(Rect(imW*8.5/36,0,imW*1/36,imH));
    Mat left_img_crop_right = left_unwarped(Rect(imW*26.5/36,0,imW*1/36,imH));
    Mat right_img_crop_left = right_unwarped(Rect(imW*8.5/36,0,imW*1/36,imH));
    Mat right_img_crop_right = right_unwarped(Rect(imW*26.5/36,0,imW*1/36,imH));
#if TEST
    imwrite("./result_picture/right_img_crop_left.jpg", right_img_crop_left);
    imwrite("./result_picture/left_img_crop_right.jpg", left_img_crop_right);
    imwrite("./result_picture/left_img_crop_left.jpg", left_img_crop_left);
    imwrite("./result_picture/right_img_crop_right.jpg", right_img_crop_right);
#endif

    Mat bleft, bright;
    fish_blend_left(bleft, left_img_crop_right, right_img_crop_left);
    fish_blend_right(bright, left_img_crop_left,right_img_crop_right);
    // Update left boundary
    bleft.copyTo(left_img_crop_right);
    bleft.copyTo(right_img_crop_left);
    // Update right boundary
    bright.copyTo(left_img_crop_left);
    bright.copyTo(right_img_crop_right);

    // BLENDING 
    Mat temp1,temp2,temp3;
    Mat part1 = left_unwarped(Rect(imW*18/36,0,imW*8.5/36,imH));
    Mat part2 = left_img_crop_right;
    Mat part3 = right_unwarped(Rect(imW*9.5/36,0,imW*17/36,imH));
    Mat part4 = left_img_crop_left;
    Mat part5 = left_unwarped(Rect(imW*9.5/36,0,imW*8.5/36,imH));
    cv::hconcat(part1,part2,temp1);
    cv::hconcat(temp1,part3,temp2);
    cv::hconcat(temp2,part4,temp3);
    cv::hconcat(temp3,part5,pano);
}

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
	string outputpath,outputpath_source,outputpath_root;
	int intersectionId,positionId;

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
    // if the dir is not existed, then creat it.
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
    outputpath_source = outputpath_root + "/source_intersection-" + to_string(intersectionId) + "-" + to_string(positionId) + ".avi";

    // initialize the (yaw pitch roll)
    float fovd = 200.0f;
    
    float cam1_yaw = 0;
    float cam1_pitch = pi*0/180;
    float cam1_roll = 0;
    float cam2_yaw = 180.5*pi/180;
    float cam2_pitch = -1.5*pi/180;
    float cam2_roll = 180.7*pi/180;
    /***************************************************************************************************
    * camera parameter 
    ***************************************************************************************************/
    float cam_mat1[3][3] = {{304.7801534402248,0.0,867.8445554617841},{0.0,305.6912573078485,625.6738350608116},{0.0,0.0,1.0}};
    float cam_dist1[4] = {0.04850296832337195,0.01454854784954367,-0.017225151508037845,0.00267053820730027};
    float cam_mat2[3][3] = {{306.5841497398096,0.0,820.9386959006093},{0.0,307.59874179252114,615.0509401922786},{0.0,0.0,1.0}};
    float cam_dist2[4] = {0.04911420665606545,0.00941556838457144,-0.011917781123122657,0.001226010291746531}; 
    Matrix3f R_l2c1(3,3),R_l2c2(3,3); 
    R_l2c1 << -0.029552813382328458,-0.9994453180676002,-0.015352114315066057, 
                0.014659576245366068,0.014923800969938616,-0.999781164549981,
                0.9994557159001571,-0.029771401669441977,0.014210404538266608;
    R_l2c2 << 0.015659396674327697,0.9996014928623236,0.02348699136139848,
                0.021552533856169342,0.023146966031160843,-0.9994997279879225,
                -0.9996450728034301,0.016157766892941516,-0.021181477494703393;
    // cam_1
    float cam1_fx = cam_mat1[0][0];
    float cam1_fy = cam_mat1[1][1];
    float cam1_u0 = cam_mat1[0][2];
    float cam1_v0 = cam_mat1[1][2];
    float p1_1 = cam_dist1[0];
    float p1_2 = cam_dist1[1];
    float p1_3 = cam_dist1[2];
    float p1_4 = cam_dist1[3];
    //cam_2
    float cam2_fx = cam_mat2[0][0];
    float cam2_fy = cam_mat2[1][1];
    float cam2_u0 = cam_mat2[0][2];
    float cam2_v0 = cam_mat2[1][2];
    float p2_1 = cam_dist2[0];
    float p2_2 = cam_dist2[1];
    float p2_3 = cam_dist2[2];
    float p2_4 = cam_dist2[3];

    //**************************************************************************
    // PreComputation 
    //**************************************************************************
    // Source image
    int Ws = 1600;
    int Hs = 1200;
    // Destination pano
    int Wd = 1800;
    int Hd = 900;
    //--------------------------------------------------------------------------
    // creat map
    //--------------------------------------------------------------------------
    Mat map1_angle_x(Hd, Wd, CV_32FC1);
    Mat map1_angle_y(Hd, Wd, CV_32FC1);
    Mat map2_angle_x(Hd, Wd, CV_32FC1);
    Mat map2_angle_y(Hd, Wd, CV_32FC1);

    //--------------------------------------------------------------------------
    // Unwarp map1_angle_x, map1_angle_y
    //--------------------------------------------------------------------------
    angle_transformation_unwarping_cam1(map1_angle_x, map1_angle_y, cam1_yaw, cam1_pitch, cam1_roll,
                                 cam1_fx,  cam1_fy,  cam1_u0,  cam1_v0,  Hs,  Ws, Hd, Wd,
                                 p1_1,  p1_2,  p1_3,  p1_4);
    angle_transformation_unwarping_cam2(map2_angle_x, map2_angle_y, cam2_yaw, cam2_pitch, cam2_roll,
                                 cam2_fx, cam2_fy, cam2_u0, cam2_v0, Hs, Ws, Hd, Wd,
                                 p2_1, p2_2, p2_3, p2_4, R_l2c1, R_l2c2);


    /***************************************************************************************************
    * thread_spilt
    ***************************************************************************************************/
    int cameraId1 = 0;
	int cameraId2 = 1;
    Mat frame1(Hs, Ws, CV_32FC1);
    Mat frame2(Hs, Ws, CV_32FC1);
    Mat pano;
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
	VideoWriter outputvideo,outputvideo_source;
	outputvideo.open(outputpath, CV_FOURCC('M', 'J', 'P', 'G'), 20.0, Size(Wd, Hd), true);
    #if capture_source
    outputvideo_source.open(outputpath_source, CV_FOURCC('M', 'J', 'P', 'G'), 20.0, Size(2*Ws, Hs), true);
    #endif
    
    while (true)
	{
    #if capture_source
        Mat frame_source;
        cv::hconcat(frame1,frame2,frame_source);
    #endif
        //--------------------------------------------------------------------------
    	// Fisheye Unwarping
   		//--------------------------------------------------------------------------
   		Mat left_unwarped, right_unwarped;
        fish_angle_regulation(map1_angle_x, map1_angle_y, frame1, left_unwarped);
        fish_angle_regulation(map2_angle_x, map2_angle_y, frame2, right_unwarped);
        //--------------------------------------------------------------------------
    	// Fisheye blending directly
   		//--------------------------------------------------------------------------
        fish_blend_directly(left_unwarped,right_unwarped,pano);
        //--------------------------------------------------------------------------
        // output and write
        //--------------------------------------------------------------------------
		outputvideo.write(pano);
    #if capture_source
        outputvideo_source.write(frame_source);
    #endif
        //--------------------------------------------------------------------------
        // debug 
        //--------------------------------------------------------------------------
		// imshow("【frame1】", frame1);
		// imwrite("./result_img/frame1.jpg",frame1);
		// imshow("【frame2】", frame2);
		// imwrite("./result_img/frame2.jpg",frame2);
        // imshow("left_unwarped",left_unwarped);
        // imwrite("./result_img/left_unwarped.jpg", left_unwarped);
        // imshow("right_unwarped",right_unwarped);
        // imwrite("./result_img/right_unwarped.jpg", right_unwarped);
        imshow("【总窗口】", pano);
        // imwrite("./result_img/pano.jpg", pano);

        pano.release();
    #if capture_source
        frame_source.release();
    #endif
        // if key get pressed, break
		if (waitKey(1) == 27)
		{
			break;
		}
    }
    outputvideo.release();
#if capture_source
    outputvideo_source.release();
#endif
	return 0;
}
