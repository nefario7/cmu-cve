# Dependencies
import os
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


# * User Input for image
input_image = input("Which image to process [carnival, smiley]? : ")


# * Reading the image as per user input
input_image_path = os.path.join("..\ps1-images", input_image)
original_image = cv.imread(input_image_path + ".jpg")
if original_image is None:
    print("Could not open or find the image: ", input_image_path)
    exit(0)


# * Window displaying Original Image
cv.namedWindow("Original Image", cv.WINDOW_NORMAL)
cv.resizeWindow("Original Image", 600, 600)
cv.imshow("Original Image", original_image)


# * Window displaying Corrected Image with a slider for adjusting gamma
cv.namedWindow("Corrected Image", cv.WINDOW_NORMAL)
cv.resizeWindow("Corrected Image", 600, 600)
cv.imshow("Corrected Image", original_image)


def on_change(val):  # Function to change the image in window as slider value changes
    gamma = val / 10
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)  # Lookup table to calculate gamma corrected pixel values
    corrected_image = cv.LUT(original_image, lookUpTable)  # Applying gamma correction on original image
    cv.imshow("Corrected Image", corrected_image)


cv.createTrackbar("Gamma (1.0 represented by 10)", "Corrected Image", 0, 100, on_change)  # Creating slider in the window
cv.waitKey(0)

gamma = int(cv.getTrackbarPos("Gamma (1.0 represented by 10)", "Corrected Image")) / 10  # Reading the final gamma chosen by user


# * Saving Corrected Image as per the gamma chosen by user
print("\nSaving the corrected image using a gamma value : ", gamma)

lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
corrected_image = cv.LUT(original_image, lookUpTable)

cv.imwrite(input_image + "_gcorrected.jpg", corrected_image)
print(f"Corrected image saved as {input_image}_gcorrected.jpg !")

cv.destroyAllWindows()
