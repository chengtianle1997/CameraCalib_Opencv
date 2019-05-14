import cv2
import numpy as np
import os

cross_corners = [7, 4]
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cross_corners[0]*cross_corners[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:cross_corners[0], 0:cross_corners[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

if __name__ == "__main__":
    file_dir = r'pic'
    pic_name = os.listdir(file_dir)


    #real_coor = np.zeros((cross_corners[0]*cross_corners[1],3),np.float32)

    for pic in pic_name:
        fname = os.path.join(file_dir,pic)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #寻找角点
        ret,corners = cv2.findChessboardCorners(gray, (cross_corners[0], cross_corners[1]), None)
        #如果找到
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
    #执行标定过程
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(mtx, "\n")
    print(dist, "\n")
    print(rvecs, "\n")
    print(tvecs, "\n")

    #测试标定结果
    img = cv2.imread('pic/test.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    #x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]
    cv2.imshow("dist", dst)
    cv2.imwrite("calibresult.jpg", dst)







