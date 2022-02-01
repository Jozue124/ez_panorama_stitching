
import os
#import pandas
import numpy as np

import cv2
import matplotlib.pyplot as plt
from PIL import Image

#variables
pts = []
descs = []
bwimgs = []
imgs = []
matchesAll = []
path = 'dataset/IMG_'
#read in images, scale them and put them in an array
for i in range(1,5):
    img = cv2.imread(path + str(i) + '.jpg')
    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 300)
    height = int(img.shape[0] * scale_percent / 300)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("img",gray)
    #cv2.waitKey()

    #find features
    #dst = cv2.cornerHarris(gray,2,3,0.04)

    #adding to arrays
    imgs.append(img)
    bwimgs.append(gray)

#now its time to find features and match points in imgs pts1->pts2->pts3->pts4 to create n-1 transformations
#orb = cv2.ORB_create()
sift = cv2.SIFT_create()

for i in range(1,4):
    img1 = imgs[i-1]
    img2 = imgs[i]
#    #feature detection returning the points and feature deescriptors
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    #print(type(des1))
    if(np.size(des1) == 0):
        cvError(0,"MatchFinder","1st descriptor empty",FILE,LINE);
    if(np.size(des2) == 0):
        cvError(0,"MatchFinder","2nd descriptor empty",FILE,LINE);
     #matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    matched = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(matched),plt.show()

    #FLANN_INDEX_KDTREE = 1
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #search_params = dict(checks = 50)
    #flann = cv2.FlannBasedMatcher(index_params, search_params)
    #matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
#    good = []
    #for m,n in matches:
    #    if m.distance < 0.7*n.distance:
    #        good.append(m)
