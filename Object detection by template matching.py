# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 18:56:11 2021

@author: abc
"""

"""

Object detection by template matching

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

#Read our image as rgb and read as gray 
img_rgb = cv2.imread("f16.jpg")
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

#Read template image
template = cv2.imread("f16_template.jpg", 0)

#find the height and width of template image
h, w = template.shape[::]

#Let's match the template image and original image
res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)

#Methods available  :['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF']

#Let's plot our match
plt.imshow(res, cmap='gray')

#Let's Extracting minimum values, maximum value and minimum location and maximum location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

#Try to draw rectangle 
top_left = min_loc  #change to max_loc for all except for TM_SQDIFF
bottom_right = (top_left[0] + w,top_left[1] + h)
cv2.rectangle(img_gray, top_left, bottom_right, 255, 2)  #255 means it's white rectangular

#Let's see our images
cv2.imshow("Original image", img_rgb)
cv2.imshow("Template image", template)
cv2.imshow("Matched image", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


##############################################################################################

#Let's take another examppe to detect multiple objects

import cv2
import numpy as np
from matplotlib import pyplot as plt

#Read our image as RGB and GRAY level
img_rgb = cv2.imread("bubbles.png")
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

#Read template image
template = cv2.imread("bubbles_template.png", 0)

#Find the height and width of template image
h , w = template.shape[::]

#Let's match the template image and original image
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

#let's plot our match
plt.imshow(res, cmap='gray')

#define threshold
threshold = 0.8
loc = np.where( res >= threshold)

#Try to draw rectangle
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

#Let's see our images
cv2.imshow("Original image ", img_gray)
cv2.imshow("Template image", template)
cv2.imshow("Matched image", img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()















