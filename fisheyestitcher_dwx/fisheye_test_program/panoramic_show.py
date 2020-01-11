# coding=utf-8
import cv2
import numpy as np
import os
import csv
import math
import colorsys
from scipy import optimize

img1_nm = "./images/images_cam1/test_"
img2_nm = "./images/images_cam2/test_"
cloud_nm = "./clouds/cloud_"

cam_mat1 = np.array([[304.7801534402248,0.0,867.8445554617841],[0.0,305.6912573078485,625.6738350608116],[0.0,0.0,1.0]])
cam_dist1 = np.array([0.04850296832337195,0.01454854784954367,-0.017225151508037845,0.00267053820730027])

cam_mat2 = np.array([[306.5841497398096,0.0,820.9386959006093],[0.0,307.59874179252114,615.0509401922786],[0.0,0.0,1.0]])
cam_dist2 = np.array([0.04911420665606545,0.00941556838457144,-0.011917781123122657,0.001226010291746531])

R_l2c1 = np.array([[-0.029552813382328458,-0.9994453180676002,-0.015352114315066057], 
                    [0.014659576245366068,0.014923800969938616,-0.999781164549981],
                    [0.9994557159001571,-0.029771401669441977,0.014210404538266608]])
R_l2c2 = np.array([[0.015659396674327697,0.9996014928623236,0.02348699136139848],
                    [0.021552533856169342,0.023146966031160843,-0.9994997279879225],
                    [-0.9996450728034301,0.016157766892941516,-0.021181477494703393]])

def point2pixel(point, cam_mat, cam_dist): #with distortion
    cam_fx = cam_mat[0,0]
    cam_fy = cam_mat[1,1]
    cam_u0 = cam_mat[0,2]
    cam_v0 = cam_mat[1,2]
    p1 = cam_dist[0]
    p2 = cam_dist[1]
    p3 = cam_dist[2]
    p4 = cam_dist[3]
    
    x = point[0]
    y = point[1]
    z = point[2]
    point_theta = math.acos(z/math.sqrt(x**2 + y**2 + z**2))
    point_theta = point_theta*(1+p1*point_theta**2+p2*point_theta**4+p3*point_theta**6+p4*point_theta**8)
    if x==0 and y==0:
        px = x*cam_fx*point_theta/math.sqrt(0.001**2 + 0.001**2)+cam_u0
        py = y*cam_fy*point_theta/math.sqrt(0.001**2 + 0.001**2)+cam_v0
        return [px, py]
    px = x*cam_fx*point_theta/math.sqrt(x**2 + y**2)+cam_u0
    py = y*cam_fy*point_theta/math.sqrt(x**2 + y**2)+cam_v0
    # px = x*cam_fx/math.sqrt(x**2 + y**2 + z**2)+cam_u0
    # py = y*cam_fy/math.sqrt(x**2 + y**2 + z**2)+cam_v0
    return [px, py]

def point2pixel2(point, cam_mat): #without distortion
    cam_fx = cam_mat[0,0]
    cam_fy = cam_mat[1,1]
    cam_u0 = cam_mat[0,2]
    cam_v0 = cam_mat[1,2]

    x = point[0]
    y = point[1]
    z = point[2]
    point_theta = math.acos(z/math.sqrt(x**2 + y**2 + z**2))
    if point_theta > np.pi:
        point_theta=np.pi*2-point_theta
    #point_theta = point_theta*(1+p1*point_theta**2+p2*point_theta**4+p3*point_theta**6+p4*point_theta**8)
    if x==0 and y==0:
        px = x*cam_fx*point_theta/math.sqrt(0.001**2 + 0.001**2)+cam_u0
        py = y*cam_fy*point_theta/math.sqrt(0.001**2 + 0.001**2)+cam_v0
        return [px, py]
    px = x*cam_fx*point_theta/math.sqrt(x**2 + y**2)+cam_u0
    py = y*cam_fy*point_theta/math.sqrt(x**2 + y**2)+cam_v0
    return [px, py]

def pixel2point(pixel, cam_mat, cam_dist):
    cam_fx = cam_mat[0,0]
    cam_fy = cam_mat[1,1]
    cam_u0 = cam_mat[0,2]
    cam_v0 = cam_mat[1,2]
    p1 = cam_dist[0]
    p2 = cam_dist[1]
    p3 = cam_dist[2]
    p4 = cam_dist[3]
    
    u=pixel[0]
    v=pixel[1]
    pixel_r=math.sqrt((u-cam_u0)**2+(v-cam_v0)**2) #*fx
    pixel_theta=pixel_r/cam_fx
    p_z=math.cos(pixel_theta)
    p_x=math.sin(pixel_theta)*(u-cam_u0)/pixel_r
    p_y=math.sin(pixel_theta)*(v-cam_v0)/pixel_r
    return [p_x,p_y,p_z]

def lola2point(longti,lati):
    p_y = math.sin(longti)
    p_z = math.cos(longti)*math.cos(lati)
    p_x = math.cos(longti)*math.sin(lati)
    return [p_x, p_y, p_z]

def eulerAnglesToRotationMatrix(ang):
    R_x = np.array([[1, 0, 0 ], [0, math.cos(ang[0]), -math.sin(ang[0]) ], [0, math.sin(ang[0]), math.cos(ang[0]) ] ]) 
    R_y = np.array([[math.cos(ang[1]), 0, math.sin(ang[1]) ], [0, 1, 0 ], [-math.sin(ang[1]), 0, math.cos(ang[1]) ] ]) 
    R_z = np.array([[math.cos(ang[2]), -math.sin(ang[2]), 0], [math.sin(ang[2]), math.cos(ang[2]), 0], [0, 0, 1] ]) 
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

def distance2RGB(distance):
    if abs(distance)<20000:
        h=distance/40000
        s=1
        v=1
        r,g,b = hsv2rgb(h,s,v)
        return (b,g,r)
    else:
        h=0.5
        s=1
        v=1
        r,g,b = hsv2rgb(h,s,v)
        return (b,g,r)

def undistort_fisheye_img():
    for i in range(20):
        img=cv2.imread("./images/cam_front/in_"+str(i+1)+".jpg")
        img_H, img_W=img.shape[:2]
        points = list()
        for v in range(img_H):
            for u in range(img_W):
                points.append(pixel2point(u,v))
        points=np.array(points)
        indxy = np.array(list(map(point2pixel, points))).T
        map_x = indxy[0].reshape(img_H, img_W).astype(np.float32)
        map_y = indxy[1].reshape(img_H, img_W).astype(np.float32)
        dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        cv2.imshow("undistorted image", dst)
        # cv2.imwrite("./result.jpg",dst)
        cv2.waitKey()
    cv2.destroyAllWindows()

def board2img():
    final_RT=np.array([0,0,0,0,0,-120],dtype=float)
    R_l2c,_=cv2.Rodrigues(final_RT[:3])
    t_l2c = final_RT[3:]
    # cloudcenters = list()
    # with open("./result/cloud_centers.csv") as f:
    #     reader=csv.reader(f)
    #     for row in reader:
    #         row=list(map(float,row))
    #         cloudcenters.append(row)
    for i in range(1):
        img=cv2.imread(img_nm+str(i+1)+".jpg")
        allpoints=list()
        with open(cloud_nm+str(i+2)+".csv") as f:
            reader=csv.reader(f)
            row_num=1
            header_row=next(reader)
            for row in reader:
                row_num+=1
                if row_num%2!=0:
                    feature=list(map(float,row[1].split("\t")[:3]))
                    allpoints.append(feature.copy())
        points=np.array(allpoints)*1000
        transd_points = np.dot(R_l2c, points.T).T + t_l2c
        points2img = list(map(point2pixel, transd_points))
        for point_id in range(len(points2img)):
            point=list(map(int,points2img[point_id]))
            point=tuple(point)
            pixelBGR = distance2RGB(transd_points[point_id,2])
            cv2.circle(img,point,1,pixelBGR,4)
        cv2.imshow("project cloud to img",img)
        cv2.waitKey()
        # break
    cv2.destroyAllWindows()


def allpoints2img():
    final_RT=np.array([np.pi/2,0,0,0,120,0],dtype=float)
    R_l2c,_=cv2.Rodrigues(final_RT[:3])
    t_l2c = final_RT[3:]
    # cloudcenters = list()
    # with open("./result/cloud_centers.csv") as f:
    #     reader=csv.reader(f)
    #     for row in reader:
    #         row=list(map(float,row))
    #         cloudcenters.append(row)
    i=4
    img=cv2.imread(img_nm+str(i)+".jpg")
    allpoints=list()
    with open(cloud_nm+str(i)+".csv") as f:
        reader=csv.reader(f)
        header_row=next(reader)
        for row in reader:
            feature=list(map(float,row[:3]))
            allpoints.append(feature.copy()) 
    points=np.array(allpoints)*1000
    transd_points = np.dot(R_l2c, points.T).T + t_l2c

    index_L=np.where(transd_points[:,2]<0)[0].tolist()
    transd_points=np.delete(transd_points,index_L,axis=0)

    points2img = list(map(point2pixel, transd_points))
    for point_id in range(len(points2img)):
        point=list(map(int,points2img[point_id]))
        point=tuple(point)
        pixelBGR = distance2RGB(transd_points[point_id,2])
        cv2.circle(img,point,1,pixelBGR,1)
    cv2.imshow("project cloud to img",img)
    cv2.imwrite("test1.jpg",img)
    cv2.waitKey()
        # break
    cv2.destroyAllWindows()


def change_view():
    ang = [0,90*np.pi/180,0]
    R = eulerAnglesToRotationMatrix(ang)
    R_inversed = np.linalg.inv(R)

    img=cv2.imread(img_nm+str(4)+".jpg")
    img_H, img_W=img.shape[:2]
    points = list()
    for v in range(img_H):
        for u in range(img_W):
            points.append(pixel2point(u,v))
    points=np.array(points)
    points_transd=(np.dot(R_inversed,points.T)).T.tolist()
    indxy = np.array(list(map(point2pixel, points_transd))).T
    map_x = indxy[0].reshape(img_H, img_W).astype(np.float32)
    map_y = indxy[1].reshape(img_H, img_W).astype(np.float32)
    dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    cv2.imshow("undistorted image", dst)
    cv2.imwrite("viewchanged.jpg",dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def img1_unfold(img):
    result_H = 900
    result_W = 1800
    points = list()
    for v in range(result_H):
        for u in range(result_W):
            lati = ((u-result_W/2)/5)*np.pi/180
            longti = ((v-result_H/2)/5)*np.pi/180
            points.append(lola2point(longti,lati))
    points=np.array(points)
    indxy = []
    for point in points:
        indxy.append(point2pixel(point, cam_mat1, cam_dist1))
    indxy = np.array(indxy).T
    map_x = indxy[0].reshape(result_H, result_W).astype(np.float32)
    map_y = indxy[1].reshape(result_H, result_W).astype(np.float32)
    dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    # cv2.imshow("unfolded image", dst)
    # cv2.imwrite("unfold_cam1_"+str(1)+".jpg",dst)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return dst


def img2_unfold(img):
    result_H = 900
    result_W = 1800
    points = list()
    for v in range(result_H):
        for u in range(result_W):
            lati = ((u-result_W/2)/5)*np.pi/180
            longti = ((v-result_H/2)/5)*np.pi/180
            points.append(lola2point(longti,lati))
    points=np.array(points)
    R_l2c2_inversed = np.linalg.inv(R_l2c2)
    R_cam12cam2 = eulerAnglesToRotationMatrix([np.pi,0,(180+0.7)*np.pi/180])
    R_tem = eulerAnglesToRotationMatrix([0, 1 *np.pi/180,0])
    R_cam12cam2 = np.dot(R_tem, R_cam12cam2)
    R_all = np.dot(R_cam12cam2, np.dot( R_l2c1, R_l2c2_inversed ))
    points_transd=(np.dot(R_all,points.T)).T.tolist()
    indxy = []
    for point in points_transd:
        indxy.append(point2pixel(point, cam_mat2, cam_dist2))
    indxy = np.array(indxy).T
    map_x = indxy[0].reshape(result_H, result_W).astype(np.float32)
    map_y = indxy[1].reshape(result_H, result_W).astype(np.float32)
    dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    # cv2.imshow("unfolded image", dst)
    # cv2.imwrite("unfold_cam2_"+str(1)+".jpg",dst)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return dst


def panoramic_show():
    result_H = 900
    result_W = 1800
    points_cam1 = list()
    for v in range(result_H):
        for u in range(result_W):
            lati = ((u-result_W/2)/5)*np.pi/180
            longti = ((v-result_H/2)/5)*np.pi/180
            points_cam1.append(lola2point(longti,lati))
    points_cam1=np.array(points_cam1)
    indxy = []
    for point in points_cam1:
        indxy.append(point2pixel(point, cam_mat1, cam_dist1))
    indxy = np.array(indxy).T
    mapx_cam1 = indxy[0].reshape(result_H, result_W).astype(np.float32)
    mapy_cam1 = indxy[1].reshape(result_H, result_W).astype(np.float32)

    points_cam2 = list()
    for v in range(result_H):
        for u in range(result_W):
            lati = ((u-result_W/2)/5)*np.pi/180
            longti = ((v-result_H/2)/5)*np.pi/180
            points_cam2.append(lola2point(longti,lati))
    points_cam2=np.array(points_cam2)
    R_l2c2_inversed = np.linalg.inv(R_l2c2)
    R_cam12cam2 = eulerAnglesToRotationMatrix([np.pi,0,(180+0.7)*np.pi/180])
    R_tem = eulerAnglesToRotationMatrix([0, 1.5 *np.pi/180,0])
    R_cam12cam2 = np.dot(R_tem, R_cam12cam2)
    R_all = np.dot(R_cam12cam2, np.dot( R_l2c1, R_l2c2_inversed ))
    points_transd=(np.dot(R_all,points_cam2.T)).T.tolist()
    indxy = []
    for point in points_transd:
        indxy.append(point2pixel(point, cam_mat2, cam_dist2))
    indxy = np.array(indxy).T
    mapx_cam2 = indxy[0].reshape(result_H, result_W).astype(np.float32)
    mapy_cam2 = indxy[1].reshape(result_H, result_W).astype(np.float32)

    cap1=cv2.VideoCapture(1)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1600 )
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200 )
    cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

    cap2=cv2.VideoCapture(2)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1600 )
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200 )
    cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    
    show_img = True
    image_id=1
    while(show_img):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        # print(frame.shape)
        ori_img1 = frame1.copy()
        ori_img2 = frame2.copy()
        dst_img1 = cv2.remap(frame1, mapx_cam1, mapy_cam1, cv2.INTER_LINEAR)
        dst_img2 = cv2.remap(frame2, mapx_cam2, mapy_cam2, cv2.INTER_LINEAR)
        dst_img1[:,:450] = dst_img2[:,900:1350]
        dst_img1[:,1350:1800] = dst_img2[:,450:900]
        cv2.imshow("stitched image", dst_img1)
        if (cv2.waitKey(1) & 0xFF) == ord('s'):
            # cv2.imwrite(img1_nm + str(image_id)+".jpg", ori_img1)#in for internal params,num=20 , ex for external params,num=29
            cv2.imwrite("img_stitch"+str(1)+".jpg",dst)
            print("image_",image_id,"is saved")
            image_id+=1
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            show_img=False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    panoramic_show()