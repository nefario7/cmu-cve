# * Dependencies
import os
import string
import random
import cv2 as cv
import numpy as np

image_data_path = r"D:\CMU\Academics\Computer Vision\PS 7\ps7-images"
image_save_path = r"D:\CMU\Academics\Computer Vision\PS 7\ps7-1"


class StereoImage:
    def __init__(self, image_name):
        self.name = image_name
        self.left = None
        self.right = None
        self.depth_params = {
            "minDisparity": 0,
            "speckleWindowSize": 200,
            "speckleRange": 2,
            "disp12MaxDiff": 2,
            "P1": 8 * 1 * 8 * 8,
            "P2": 32 * 1 * 8 * 8,
        }

    def __set_depth_params(self):
        if self.name == "plant":
            p = [96, 5, 9]
        elif self.name == "baby":
            p = [96, 7, 3]
        elif self.name == "ball":
            p = [100, 9, 3]
        else:
            p = [64, 5, 3]
        self.depth_params["numDisparities"] = p[0]
        self.depth_params["blockSize"] = p[1]
        self.depth_params["uniquenessRatio"] = p[2]
        if len(p) > 3:
            self.depth_params["preFilterCap"] = p[3]

    def __write_ply(self, fn, verts, colors):
        ply_header = """ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
            """
        verts = verts.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        verts = np.hstack([verts, colors])
        with open(fn, "wb") as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode("utf-8"))
            np.savetxt(f, verts, fmt="%f %f %f %d %d %d ")

    def read_images(self):
        try:
            image_data = os.listdir(image_data_path)
            image_exts = np.unique([img.split(".")[1] for img in image_data if img.split("-")[0] == self.name])
            ext = image_exts[0]
            self.left = cv.imread(os.path.join(image_data_path, self.name + "-left." + ext))
            self.right = cv.imread(os.path.join(image_data_path, self.name + "-right." + ext))
        except:
            print("\nThe entered image could not be found!")
            exit(0)

        # if self.left.shape[0] > 1280 or self.left.shape[1] > 1280:
        #     ratio = self.left.shape[0] / self.left.shape[1]
        #     size = (640, 480) if ratio < 1 else (480, 640)
        #     self.left = cv.resize(self.left, size)
        #     self.right = cv.resize(self.right, size)

        # show(self.left)
        # show(self.right)

    def disparity_map(self, save=False, blur=True):
        if blur:
            left_image = cv.GaussianBlur(self.left, (11, 11), cv.BORDER_DEFAULT)
            right_image = cv.GaussianBlur(self.right, (11, 11), cv.BORDER_DEFAULT)
        else:
            left_image = self.left
            right_image = self.right
        self.__set_depth_params()
        stereo = cv.StereoSGBM_create(**self.depth_params)
        disparity = stereo.compute(left_image, right_image)

        cv.normalize(disparity, disparity, alpha=255, beta=0, norm_type=cv.NORM_MINMAX)
        self.disparity = np.uint8(disparity)

        if save:
            disparity_path = os.path.join(image_save_path, self.name + "-disparity.png")
            cv.imwrite(disparity_path, disparity)
            print("\nDisparity map saved at : ", disparity_path)

    def point_cloud(self, save=False):
        h = self.left.shape[0]
        w = self.left.shape[1]
        rev_proj_matrix = np.float32([[1, 0, 0, -0.5 * w], [0, -1, 0, 0.5 * h], [0, 0, 0, -1.2 * w], [0, 0, 1, 0]])  # to store the output

        points = cv.reprojectImageTo3D(self.disparity, rev_proj_matrix)
        colors = cv.cvtColor(self.left, cv.COLOR_BGR2RGB)

        mask = self.disparity > self.disparity.min()
        out_points = points[mask]
        out_colors = colors[mask]

        if save:
            point_path = os.path.join(image_save_path, self.name + ".ply")
            self.__write_ply(point_path, out_points, out_colors)
            print("\nPoint cloud data saved at : ", point_path)


# * Function to display image in a window
def show(img, name=None):
    if name is None:
        letters = string.ascii_lowercase
        name = "".join(random.choice(letters) for i in range(10))
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, img)
    cv.waitKey(0)


if __name__ == "__main__":
    # * Reading the list of available images in the
    image_data = os.listdir(image_data_path)
    image_data = np.unique([img.split("-")[0] for img in image_data])
    print("List of stereo pairs = ", image_data)

    image_name = input("Enter the name of stereo pair to get point-cloud : ")

    # * Initialize a StereoImage object
    image = StereoImage(image_name)
    image.read_images()
    image.disparity_map(save=True)  # Generate and save a disparity map
    image.point_cloud(save=True)  # Generate and save point cloud data
