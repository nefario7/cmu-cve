import os
import cv2 as cv
import numpy as np
import imutils

import random
import string
import math

colors = {"ring": (255, 0, 0), "spade": (0, 0, 255), "washer": (0, 255, 0), "internal": (255, 0, 150), "external": (0, 250, 255)}

# * Function to display image in a window
def show(img, name=None):
    if name is None:
        letters = string.ascii_lowercase
        name = "".join(random.choice(letters) for i in range(10))
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, img)
    cv.waitKey(0)


# * Function to calculate solidity of contour
def solidity(ctr):
    area = cv.contourArea(ctr)
    hull = cv.convexHull(ctr)
    hull_area = cv.contourArea(hull)
    solidity = float(area) / hull_area
    return solidity


# * Function to classify contour into different parts
def check_contour(contours, hrc, i):
    ctr = contours[i]
    boundary = 0
    fill = 0
    part = ""

    # ? Spade
    if hrc[3] == boundary and hrc[2] == -1:
        part = "spade"
        fill = 1
    elif hrc[2] != -1:
        # ? Ring
        _, dims, _ = cv.minAreaRect(ctr)
        aspect_ratio = dims[0] / dims[1]
        thres = 1.2
        if aspect_ratio > thres or 1 / aspect_ratio > thres:
            part = "ring"
        else:
            if solidity(ctr) < 0.98:
                part = "external"
            else:
                if math.ceil(solidity(contours[hrc[2]]) * 10) == 10:
                    part = "washer"
                else:
                    part = "internal"
    return part, fill


if __name__ == "__main__":
    # * Reading the image
    image_name = "all-parts.png"
    original_image = cv.imread(image_name)
    if original_image is None:
        print("\nThe entered image could not be found!")
        exit(0)

    # Final processed image
    colored_image = original_image.copy()

    # Convert to grayscale
    image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    image = cv.GaussianBlur(image, (9, 9), 9)

    # Conver to binary
    _, image = cv.threshold(image, 50, 255, cv.THRESH_BINARY)

    # Morphological Operations
    for _ in range(2):
        image = cv.dilate(image, None)
    image = cv.erode(image, None)

    # Contour Detection
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Contour Assignment
    for i in range(1, len(contours)):
        part, fill = check_contour(contours, hierarchy[0][i], i)
        if len(part) != 0:
            if fill:
                cv.drawContours(colored_image, contours, i, colors[part], 5)
                cv.drawContours(colored_image, contours, i, colors[part], -1)
            else:
                cv.drawContours(colored_image, contours, i, colors[part], 5)
                cv.drawContours(colored_image, contours, i, colors[part], -1)
                # cv.drawContours(colored_image, contours, hierarchy[0][i][2], (255, 255, 255), -1)

    cv.imwrite("all-parts-output.png", colored_image)
    cv.destroyAllWindows()
