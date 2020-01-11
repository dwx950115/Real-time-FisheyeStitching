import cv2
import numpy as np
import os

img1_nm = "./images_cam1/all_"
img2_nm = "./images_cam2/all_"

def main():
    cap1=cv2.VideoCapture(0)
    cap2=cv2.VideoCapture(1)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1600 )
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200 )
    cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1600 )
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200 )
    cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    image_id=1
    both_cams=True

    ###collect both images
    while(both_cams):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        #frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
        ori_img1=frame1.copy()
        ori_img2=frame2.copy()
        frame1,_=get_corners(frame1)
        frame2,_=get_corners(frame2)
        cv2.imshow('Camera_1',frame1)
        cv2.imshow('Camera_2',frame2)
        if (cv2.waitKey(1) & 0xFF) == ord('s'):
            cv2.imwrite(img1_nm +str(image_id)+".jpg",ori_img1)#num=29
            cv2.imwrite(img2_nm +str(image_id)+".jpg",ori_img2)#num=29
            #cv2.imwrite(os.path.join("./images","ex_L_"+str(image_id)+".jpg"),ori_img[:,0:640])#num=29
            #cv2.imwrite(os.path.join("./images","ex_R_"+str(image_id)+".jpg"),ori_img[:,640:1280])#num=29
            print("image_id",image_id,"is saved")
            image_id+=1
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            both_cams=False

    cap1.release()
    # cap2.release()
    cv2.destroyAllWindows()




def get_corners(img):
    size=(6-1,8-1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, size,flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
    #flags=cv2.cv.CV_CALIB_CB_ADAPTIVE_THRESH + cv2.cv.CV_CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    if not ret:
        corners=None
        return img,corners
    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                     30, 0.001)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), stop_criteria)
    corners_reshaped=corners.reshape((size[1],size[0],2))
    corners_reshaped=np.flip(corners_reshaped,1)
    corners=corners_reshaped.reshape((size[0]*size[1],1,2))
    cv2.drawChessboardCorners(img, size, corners, ret)
    corner_center=tuple(corners[17].astype(np.int).flatten(0).tolist())
    cv2.circle(img,corner_center,10,(255,255,0),1)
    return img,corners
    
    
def show_corners_in_image():
    for i in range(29):
        img= cv2.imread(os.path.join("./images","ex_L_"+str(i+1)+".jpg"))
        img,_=get_corners(img)
        print("img_"+str(i+1)+" has detected corners!")
        print(img.shape[::-1])
        cv2.imshow('Image',img)
        cv2.waitKey()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
    # show_corners_in_image()
