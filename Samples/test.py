# -*- coding:utf-8 -*-
# mycv.py

import sys

import cv2, os
import numpy as np
import math
from math import cos,sin
import pickle
from os import listdir
# from cv2 import *



def subimage_cv(image, centre, theta, width, height):
   output_image = cv2.cv.CreateImage((width, height), image.depth, image.nChannels)
   mapping = np.array([[np.cos(theta), -np.sin(theta), centre[0]],
                       [np.sin(theta), np.cos(theta), centre[1]]])
   map_matrix_cv = cv2.cv.fromarray(mapping)
   cv2.cv.GetQuadrangleSubPix(image, output_image, map_matrix_cv)
   return output_image

def subimage_cv2(image, center, theta, width, height,flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE):
    # http://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
    # http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    # http://blogs.candoerz.com/question/67841/align-x-ray-images-find-rotation-rotate-and-crop.aspx
    theta *= 3.14159 / 180 # convert to rad

    v_x = (cos(theta), sin(theta))
    v_y = (-sin(theta), cos(theta))
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    width = int(math.ceil(width))
    height = int(math.ceil(height))

    return cv2.warpAffine(image,mapping,(width, height),flags=flags,borderMode=borderMode)
def rotate_clockwise(img):
    return cv2.flip(cv2.transpose(img),1)


if __name__ == '__main__':

    filenames = ['test.png']
    filepath = ''
    outputpath = ""

    MIN_AREA_RATIO = 0.01
    SCALE_RATIO = 1
    margin_ratio = 0.02
    TIMEOUT = 0

    cv2.namedWindow("ShowImage", cv2.WINDOW_NORMAL)
    cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)

    for sn, filename in enumerate(filenames):
        if not (".png" in filename or ".jpg" in filename or ".jpeg" in filename):
            continue

        fullname = os.path.join(filepath,filename)
        #fullname = filename

        # img_org = cv2.imread(fullname)
        img = cv2.imread(fullname)
        img = rotate_clockwise(img)

        a = img.shape[0]
        b = img.shape[1]

        print a,b,img.shape
        img = img[int(margin_ratio*a):int((1-margin_ratio)*a),
                   int(margin_ratio * b):int((1-margin_ratio) * b)]  #int(0.05 * b):int(0.95 * b)

        img_1 = img.copy()

        MIN_AREA = img_1.shape[0] * img_1.shape[1] * MIN_AREA_RATIO
        # dst = img
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # blurred = cv2.blur(gray,(10,10))

        im = gray
        # ret, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ret, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY )
        # determines the threshold automatically from the image using Otsu's method

        # construct a closing kernel and apply it to the thresholded image
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("thresh", thresh)
        cv2.resizeWindow("thresh",
                         int(thresh.shape[1]*SCALE_RATIO),
                         int(thresh.shape[0] * SCALE_RATIO))

        contours, hierarchy = cv2.findContours(thresh,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key = cv2.contourArea, reverse = True)

        t = []
        for idx,c  in enumerate(cnts):
            if cv2.contourArea(c) < MIN_AREA:
                break
            # compute the rotated bounding box of the largest contour
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.cv.BoxPoints(rect))
            cv2.drawContours(img, [box], -1, (0, 255, 0), min(255,max(3,int(a*0.005))))
            # http://felix.abecassis.me/2011/10/opencv-bounding-box-skew-angle/
            # http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
            c_x, c_y = rect[0]
            c_x = int(c_x)
            c_y = int(c_y)
            cv2.circle(img, (c_x,c_y), min(255,max(3,int(a*0.01))), (0, 0, 255), -1)
            # ===============================================

            # compute the rotated bounding box of the largest contour
            rect = cv2.minAreaRect(c)  # cv2.minAreaRect() is ((x, y), (w, h), angle)
            box = np.int0(cv2.cv.BoxPoints(rect))
            angle = rect[2]
            if (angle < -45.):
                angle += 90.

            center = rect[0]
            width, height = rect[1]
            if angle > 0:
                height, width = rect[1]
            print "Angle", angle, rect
            print box

            cv2.circle(img, (box[0][0],box[0][1]), 5, (255, 0, 0), -1)

            flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
            flags = cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC
            flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4
            borderMode = cv2.BORDER_REPLICATE
            #borderMode = cv2.BORDER_WRAP
            borderMode = cv2.BORDER_DEFAULT
            # borderMode = cv2.BORDER_TRANSPARENT

            # http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#void resize(InputArray src, OutputArray dst, Size dsize, double fx, double fy, int interpolation)
            # http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#void warpAffine(InputArray src, OutputArray dst, InputArray M, Size dsize, int flags, int borderMode, const Scalar& borderValue)
            t = subimage_cv2(img_1, center, angle, width, height,
                             flags=flags,
                             borderMode=borderMode)

            # cv2.namedWindow("ShowImage2")
            # cv2.imshow("ShowImage2", t)
            #tt = np.array([], dtype=t.dtype)
            # t = rotate_clockwise(t)
            outputname = os.path.join(outputpath,filename.split('.')[0] + " "+ str(idx) +".jpg")
            cv2.imwrite(outputname, t)

        cv2.imshow ("ShowImage", img)
        cv2.resizeWindow("ShowImage",
                         int(img.shape[1] * SCALE_RATIO),
                         int(img.shape[0] * SCALE_RATIO))

    # ====================
        k = cv2.waitKey (TIMEOUT*1000)
        print "Key", k
        if  k == 113: # q
            break

    # cv2.imwrite("D:\\cat2.jpg", img)
    cv2.destroyAllWindows()
