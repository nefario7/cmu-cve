import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# * Getting user input
image_folder = "ps2-images"
print("Available images in folder : ", os.listdir(image_folder))

image_name = str(input("Enter the image that you would like to process : "))
image_path = os.path.join(image_folder, image_name + ".png")
image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

if image is None:
    print("\nThe entered image could not be found!")
    exit(0)


# * Display Image
cv.namedWindow("Input Image", cv.WINDOW_NORMAL)
cv.resizeWindow("Input Image", 900, 600)
cv.imshow("Input Image", image)


# * Get minimum and maximum pixel value
min_pix = np.amin(image)
max_pix = np.amax(image)
range_pix = max_pix - min_pix
min_indices = np.where(image == [min_pix])
max_indices = np.where(image == [max_pix])
print(min_indices)
max_coordinate = (int(np.mean(max_indices[1])), int(np.mean(max_indices[0])))


# * Colormap definition
def pcolor(p):
    m = 4 / range_pix  # Unit slope
    s = m * p  # Unit slope x Pixel value
    if p <= range_pix / 4:
        return [255, 255 * s, 0]  # 255 multiplied to unit slope for scaling to [0,255]
    elif p > range_pix / 4 and p <= range_pix / 2:
        return [255 * (2 - s), 255, 0]
    elif p > range_pix / 2 and p <= 3 * range_pix / 4:
        return [0, 255, 255 * (s - 2)]
    elif p > 3 * range_pix / 4 and p < range_pix
        return [0, 255 * (4 - s), 255]
    elif p >= range_pix:
        return [0, 0, 255]


# * Plot to check the colormap
# pixel = np.arange(min_pix, max_pix)
# vpcolor = np.vectorize(pcolor)
# vals = np.empty((range_pix, 3))
# for i in pixel:
#     vals[i, 0] = pcolor(i)[0]
#     vals[i, 1] = pcolor(i)[1]
#     vals[i, 2] = pcolor(i)[2]

# print(vals.shape)
# plt.plot(pixel, vals[:, 0], "b")
# plt.plot(pixel, vals[:, 1], "g")
# plt.plot(pixel, vals[:, 2], "r")
# plt.show()

# * Generate Pseudocolour map
pseudo_lut = np.empty((256, 1, 3), np.uint8)
for i in range(256):
    color = pcolor(i)
    pseudo_lut[i, 0, 0] = int(color[0])
    pseudo_lut[i, 0, 1] = int(color[1])
    pseudo_lut[i, 0, 2] = int(color[2])


# * Apply Pseudocolours on the image
image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
pseudo_image = cv.LUT(image, pseudo_lut)


radius = int(image.shape[0] * 0.05)
thickness = 1 if image.shape[0] < 600 else 2
pseudo_image = cv.circle(pseudo_image, max_coordinate, radius, (255, 255, 255), thickness)
pseudo_image = cv.drawMarker(pseudo_image, max_coordinate, (255, 255, 255), cv.MARKER_CROSS, radius, thickness, 8)


# * Display new image
cv.namedWindow("Pseudocolor Image", cv.WINDOW_NORMAL)
cv.resizeWindow("Pseudocolor Image", 900, 600)
cv.imshow("Pseudocolor Image", pseudo_image)
cv.imwrite(os.path.join("ps2-1", image_name + "-color" + ".png"), pseudo_image)

cv.waitKey(0)
cv.destroyAllWindows()
