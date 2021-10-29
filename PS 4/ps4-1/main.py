import os
import json
import numpy as np
import cv2 as cv
from tqdm import tqdm

from utility import *

if __name__ == "__main__":
    image_data_path = r"ps4-images"
    image_save_path = r"ps4-1"

    # * Reading the list of available images in the
    image_data = os.listdir(image_data_path)
    image_data = [name.split("-")[0] for name in image_data]
    image_data = np.unique(np.array(image_data))

    print("List of images = ", image_data)

    image_name = input("Enter the name of the image to process : ")
    # image_name = "door"

    image_path = os.path.join(image_data_path, image_name)
    if image_name == "wall":
        extension = ".png"
    else:
        extension = ".jpg"
    imageL = cv.imread(os.path.join(image_data_path, image_name + "-left" + extension))
    imageC = cv.imread(os.path.join(image_data_path, image_name + "-center" + extension))
    imageR = cv.imread(os.path.join(image_data_path, image_name + "-right" + extension))

    # Check to see if image is NoneType
    if any(img is None for img in [imageL, imageC, imageR]):
        print("Image could not be read!")
        exit(0)

    jerry = MouseClick(imageR, imageC, imageL, image_name)

    i = "n"
    if os.path.isfile(image_name + "-points.json"):
        i = input("Use saved points from points.json? (y/n) : ")

    if i == "y" or i == "Y":
        try:
            jerry.points = load_point_data(image_name)
            combined_image = jerry.combine()
        except FileNotFoundError:
            print("Points data could not be found for -", image_name, "!")
            exit
    else:
        cv.namedWindow("left", cv.WINDOW_NORMAL)
        cv.moveWindow("left", 10, 100)
        cv.namedWindow("center", cv.WINDOW_NORMAL)
        cv.moveWindow("center", 640, 200)
        cv.namedWindow("right", cv.WINDOW_NORMAL)
        cv.moveWindow("right", 1280, 300)

        cv.imshow("left", imageL.copy())
        cv.imshow("center", imageC.copy())
        cv.imshow("right", imageR.copy())

        jerry.pick_points()

    cv.waitKey(0)
    cv.destroyAllWindows()

