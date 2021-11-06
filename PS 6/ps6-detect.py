import os
import cv2 as cv
import numpy as np
import imutils

import string
import random

# * Function to display image in a window
def show(img, name=None):
    if name is None:
        letters = string.ascii_lowercase
        name = "".join(random.choice(letters) for i in range(10))
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, img)
    cv.waitKey(0)


# * Function to extract template from the original image
def template_extraction(reference):
    template = cv.imread("spade-template.png")
    if template is None:
        reference = cv.cvtColor(reference, cv.COLOR_GRAY2BGR)
        template_data = cv.selectROI(reference)
        template = reference[
            int(template_data[1]) : int(template_data[1] + template_data[3]),
            int(template_data[0]) : int(template_data[0] + template_data[2]),
        ]
        w = template_data[3] - template_data[1]
        h = template_data[2] - template_data[0]

        template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
        # To rotate the templat to vertical if needed
        # contour, _ = cv.findContours(template, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        # c, _, a = cv.minAreaRect(contour[0])

        # M = cv.getRotationMatrix2D((template.shape[0] / 2, template.shape[1] / 2), a - 90, 1.0)
        # template = cv.warpAffine(template, M, (int(w), int(h)), borderValue=(255, 255, 255))

        cv.imwrite("spade-template.png", template)
    else:
        template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    return template


if __name__ == "__main__":
    # * Reading the image
    image_name = "spade-terminal.png"
    image_spade = cv.imread(image_name)
    if image_spade is None:
        print("\nThe entered image could not be found!")
        exit(0)

    # Final processed image
    coloured_image = image_spade.copy()

    # Convert to grayscale
    image = cv.cvtColor(image_spade, cv.COLOR_BGR2GRAY)

    # Convert to binary
    _, image = cv.threshold(image, 50, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    image = cv.dilate(image, kernel, iterations=1)
    image = cv.erode(image, kernel, iterations=1)

    # Template Matching
    template = template_extraction(image)
    main_contours, _ = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    template_contour, _ = cv.findContours(template, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for i in range(len(main_contours) - 1):
        match = cv.matchShapes(main_contours[i], template_contour[0], cv.CONTOURS_MATCH_I2, 0)
        if match > 1:
            coloured_image = cv.drawContours(coloured_image, main_contours, i, (0, 0, 255), -1)

    cv.imwrite("spade-terminal-output.png", coloured_image)
    print("Output saved!")
