import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from tqdm import tqdm
from utility import *

# * DETECTORS

# * Sobel Edge Detector Implementation
def sobel_filter(input_image):
    sobelkernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)

    vertical_edges = apply_convolution(input_image, sobelkernel)  # First derivative in Horizontal Direction
    horizontal_edges = apply_convolution(input_image, sobelkernel.T)  # First derivative in Vertical Direction

    edge_gradient = np.sqrt(np.square(vertical_edges) + np.square(horizontal_edges))
    edge_gradient *= 255.0 / edge_gradient.max()
    # gradient_angle = np.arctan(horizontal_edges / vertical_edges)
    # gradient_angle = (gradient_angle * 180 / np.pi) + 180

    edge_gradient = 255 - edge_gradient

    # cv.namedWindow("Sobel Filter - Detected Edges")
    # cv.imshow("Sobel Filter - Detected Edges", edge_gradient)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return edge_gradient


# * Canny Edge Detector Implementation
def canny_edge_detector(input_image):
    def change(val):
        pass

    cv.namedWindow("Canny Edge Detections", cv.WINDOW_NORMAL)
    cv.createTrackbar("Threshold 1", "Canny Edge Detections", 0, 2000, change)
    cv.createTrackbar("Threshold 2", "Canny Edge Detections", 0, 2000, change)
    cv.createTrackbar("Aperture", "Canny Edge Detections", 0, 2, change)
    l2norm = "L2 Norm \n0 : OFF \n1 : ON"
    cv.createTrackbar(l2norm, "Canny Edge Detections", 0, 1, change)

    while True:
        thres1 = cv.getTrackbarPos("Threshold 1", "Canny Edge Detections")
        thres2 = cv.getTrackbarPos("Threshold 2", "Canny Edge Detections")
        apert = cv.getTrackbarPos("Aperture", "Canny Edge Detections")
        gradient = cv.getTrackbarPos(l2norm, "Canny Edge Detections")
        output = cv.Canny(input_image, thres1, thres2, apertureSize=2 * (apert + 1) + 1, L2gradient=gradient)
        output = cv.bitwise_not(output)
        cv.imshow("Canny Edge Detections", output)

        if cv.waitKey(1) & 0xFF == ord("e"):
            break
    cv.destroyAllWindows()
    print(thres1, thres2, apert, gradient)
    return output, [thres1, thres1, apert, gradient]


# def canny_edge(input_image):
#     grads, angles = sobel_filter(input_image)
#     nonmax = np.zeros(input_image.shape)

#     for row in tqdm(range(input_image.shape[0] - 1)):
#         for col in range(input_image.shape[1] - 1):
#             gradient = grads[row][col]
#             direction = angles[row][col]

#             ranges = np.linspace(np.pi / 8, 2 * np.pi + np.pi / 8, 8, endpoint=False)
#             ranges *= 180 / np.pi
#             # [ 22.5  67.5 112.5 157.5 202.5 247.5 292.5 337.5]

#             if (0 <= direction <= 22.5) or (22.5 <= direction <= 360) or (157.5 <= direction <= 202.5):
#                 before = grads[row, col - 1]
#                 after = grads[row, col + 1]
#             elif (22.5 <= direction <= 67.5) or (292.5 <= direction <= 337.5):
#                 before = grads[row + 1, col - 1]
#                 after = grads[row - 1, col + 1]
#             elif (67.5 <= direction <= 112.5) or (247.5 <= direction <= 292.5):
#                 before = grads[row - 1, col]
#                 after = grads[row + 1, col]
#             elif (112.5 <= direction <= 157.5) or (202.5 <= direction <= 247.5):
#                 before = grads[row - 1, col - 1]
#                 after = grads[row + 1, col + 1]

#             if gradient >= before and gradient >= after:
#                 nonmax[row, image] = gradient

#     # weak_ids = np.zeros(input_image.shape)
#     # strong_ids = np.zeros(input_image.shape)
#     # ids = np.zeros(input_image.shape)
#     # for row in tqdm(range(input_image.shape[0])):
#     #     for col in range(input_image.shape[1]):
#     #         gradient = grads[row][col]
#     #         if grad_mag < weak_th:
#     #             mag[col, row] = 0
#     #         elif strong_th > grad_mag >= weak_th:
#     #             ids[col, row] = 1
#     #         else:
#     #             ids[col, row] = 2

