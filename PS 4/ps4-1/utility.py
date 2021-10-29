# import the necessary packages
import os
import sys
import json
import argparse
import numpy as np
import cv2 as cv


def save_point_data(points, filename):
    with open(filename + "-points.json", "w") as outfile:
        json.dump(points, outfile)


def load_point_data(filename):
    with open(filename + "-points.json") as file:
        data = json.load(file)
    print(data)
    return data


class MouseClick:
    # * Constructor
    def __init__(self, imageR, imageC, imageL, image_name):
        self.name = image_name
        self.image = dict()
        self.image["right"] = imageR
        self.image["center"] = imageC
        self.image["left"] = imageL

        self.h = self.image["center"].shape[0]
        self.w = self.image["center"].shape[1]

        self.points = {"left": [], "right": [], "center": {"left": [], "right": []}}

    # * Call the mouse_click function to start selecting points
    def pick_points(self):
        print("\nPick points on right image!")
        cv.setMouseCallback("right", self.click, param=["right", None])

    # * Callback function for setMouseCallback
    def click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONUP:
            self.mouse_pick(x, y, param[0], param[1])

    # * Process mouse clicks and show on images
    def mouse_pick(self, x, y, side, relation=None):
        src = self.image[side]
        dst = src.copy()

        print(side, x, y)
        if relation is None:
            self.points[side].append((x, y))
            temp_points = self.points[side]
        else:
            self.points[side][relation].append((x, y))
            temp_points = self.points[side][relation]

        if side == "left" or relation == "left":
            col = (255, 0, 0)
        elif side == "right" or relation == "right":
            col = (0, 0, 255)

        for i in range(len(temp_points)):
            dst = cv.circle(dst, temp_points[i], 5, col, 2)
            dst = cv.putText(dst, str(i), (temp_points[i][0] + 10, temp_points[i][1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1, col, 2)

        if side == "center" and relation == "left":
            for i in range(len(self.points["center"]["right"])):
                dst = cv.circle(dst, self.points["center"]["right"][i], 5, (0, 0, 255), 2)
                dst = cv.putText(
                    dst,
                    str(i),
                    (self.points["center"]["right"][i][0] + 10, self.points["center"]["right"][i][1] - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
        # please make sure when idx == 3, you need to show red color circle in dst
        # this example erases red circle

        cv.imshow(side, dst)
        cv.waitKey(1)

        if len(temp_points) >= 4:
            print(side, relation, temp_points)
            i = input("Is it OK? (y/n) : ")
            if i == "y" or i == "Y":
                if side == "right":
                    print("\nPick points on center image corresponding to right!")
                    cv.setMouseCallback("center", self.click, param=["center", "right"])

                elif side == "center" and relation == "right":
                    print("\nPick points on left image!")
                    cv.setMouseCallback("left", self.click, param=["left", None])

                elif side == "left":
                    print("\nPick points on center image corresponding to left!")
                    cv.setMouseCallback("center", self.click, param=["center", "left"])

                elif side == "center" and relation == "left":
                    print("Point picking complete! Saving picked points in points.json.")
                    save_point_data(self.points, self.name)
                    print("Combining images!")
                    self.combine()
                else:
                    print("Something is wrong!")
            else:
                if relation is None:
                    self.points[side] = []
                else:
                    self.points[side][relation] = []
                dst = src.copy()
                cv.imshow(side, dst)

    # * Convert points into numpy arrays
    def np_points(self, side):
        src_pnts = np.empty([4, 2], np.float32)
        dst_pnts = np.empty([4, 2], np.float32)
        for i in range(4):
            src_pnts[i][0] = float(self.points[side][i][0])
            src_pnts[i][1] = float(self.points[side][i][1])
            dst_pnts[i][0] = float(self.points["center"][side][i][0] + self.w)
            dst_pnts[i][1] = float(self.points["center"][side][i][1] + self.h)

        return src_pnts, dst_pnts

    # * Combine the left, center and right images into one panorama
    def combine(self, loaded_data=None):
        if loaded_data is not None:
            self.points = loaded_data.copy()

        result = cv.copyMakeBorder(self.image["center"], self.h, self.h, self.w, self.w, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])

        print(self.image["left"].shape, self.image["center"].shape, self.image["right"].shape, result.shape)

        # * Center
        cng = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
        th, mask_c = cv.threshold(cng, 1, 255, cv.THRESH_BINARY)
        mask_c = mask_c / 255

        # * Right side
        src_pnts, dst_pnts = self.np_points("right")
        M = cv.getPerspectiveTransform(src_pnts, dst_pnts)
        rn = cv.warpPerspective(self.image["right"], M, (self.w * 3, self.h * 3))
        rng = cv.cvtColor(rn, cv.COLOR_BGR2GRAY)
        th, mask_r = cv.threshold(rng, 1, 255, cv.THRESH_BINARY)
        mask_r = mask_r / 255
        # cv.imwrite("mask_r.png", mask_r)

        # * Left side
        src_pnts, dst_pnts = self.np_points("left")
        M = cv.getPerspectiveTransform(src_pnts, dst_pnts)
        ln = cv.warpPerspective(self.image["left"], M, (self.w * 3, self.h * 3))
        lng = cv.cvtColor(ln, cv.COLOR_BGR2GRAY)
        th, mask_l = cv.threshold(lng, 1, 255, cv.THRESH_BINARY)
        mask_l = mask_l / 255

        # alpha blending
        # mask element: number of pictures at that coordinate
        mask = np.array(mask_c + mask_l + mask_r, float)

        # alpha blending weight
        ag = np.full(mask.shape, 0.0, dtype=float)
        ag = 1.0 / np.maximum(1, mask)

        # generate result image from 3 images + alpha weight
        result[:, :, 0] = result[:, :, 0] * ag[:, :] + ln[:, :, 0] * ag[:, :] + rn[:, :, 0] * ag[:, :]
        result[:, :, 1] = result[:, :, 1] * ag[:, :] + ln[:, :, 1] * ag[:, :] + rn[:, :, 1] * ag[:, :]
        result[:, :, 2] = result[:, :, 2] * ag[:, :] + ln[:, :, 2] * ag[:, :] + rn[:, :, 2] * ag[:, :]

        cv.namedWindow("result", cv.WINDOW_NORMAL)
        cv.imshow("result", result)
        cv.imwrite(os.path.join(r"ps4-1", self.name + "-stitched.png"), result)

