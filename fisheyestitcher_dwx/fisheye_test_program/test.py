# coding=utf-8
import cv2
import numpy as np
import os
import csv
import math
import colorsys
from scipy import optimize

img_nm = "./images/test_"
cloud_nm = "./clouds/cloud_"

cam_mat2 = np.array([[305.8103327739443, 0.0, 868.4087733419921], [0.0, 306.2007353016409, 628.2346880123545], [0.0, 0.0, 1.0]])
cam_dist2 = np.array([[0.05474630002513208], [-0.003933657882507959], [0.03314157850495621], [-0.03770454084099611]])
cam_mat1 = np.array([[306.2179554207244, 0.0, 827.6396074947415], [0.0, 306.4032729038873, 612.7686375589241], [0.0, 0.0, 1.0]])
cam_dist1 = np.array([[0.061469387879628086], [-0.018668965639305285], [0.03781464752574166], [-0.02677018297407956]])

cam_fx = cam_mat2[0][0]
cam_fy = cam_mat2[1][1]
cam_u0 = cam_mat2[0][2]
cam_v0 = cam_mat2[1][2]
p1 = cam_dist2[0][0]
p2 = cam_dist2[1][0]
p3 = cam_dist2[2][0]
p4 = cam_dist2[3][0]


def point2pixel(point):
    global cam_fx,cam_fy,cam_u0,cam_v0,p1,p2,p3,p4
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
    # px = x*cam_fx/math.sqrt(x**2 + y**2 + z**2)+cam_u0
    # py = y*cam_fy/math.sqrt(x**2 + y**2 + z**2)+cam_v0
    return [px, py]

def point2pixel2(point):
    global cam_fx,cam_fy,cam_u0,cam_v0,p1,p2,p3,p4
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
    # px = x*cam_fx/math.sqrt(x**2 + y**2 + z**2)+cam_u0
    # py = y*cam_fy/math.sqrt(x**2 + y**2 + z**2)+cam_v0
    return [px, py]

def pixel2point(u,v):
    global cam_fx,cam_fy,cam_u0,cam_v0,p1,p2,p3,p4
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
    cv2.waitKey()
    cv2.destroyAllWindows()


def img_unfold():
    result_H = 900
    result_W = 1800
    
    img=cv2.imread(img_nm+str(4)+".jpg")
    points = list()
    for v in range(result_H):
        for u in range(result_W):
            lati = ((u-result_W/2)/5)*np.pi/180
            longti = ((v-result_H/2)/5)*np.pi/180
            points.append(lola2point(longti,lati))
    points=np.array(points)
    indxy = np.array(list(map(point2pixel, points))).T
    map_x = indxy[0].reshape(result_H, result_W).astype(np.float32)
    map_y = indxy[1].reshape(result_H, result_W).astype(np.float32)
    dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    cv2.imshow("unfolded image", dst)
    cv2.imwrite("img_unfold"+str(4)+".jpg",dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

def img_stitch():
    img1=cv2.imread("img_unfold"+str(4)+".jpg")
    img2=cv2.imread("img_unfold"+str(5)+".jpg")
    img1[:,:450] = img2[:,900:1350]
    img1[:,1350:1800] = img2[:,450:900]
    dst = img1
    cv2.imshow("stitched image", dst)
    cv2.imwrite("img_stitch"+str(4)+".jpg",dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # board2img()
    # allpoints2img()
    # change_view()
    # img_unfold()
    img_stitch()