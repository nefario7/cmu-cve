import os
import cv2 as cv
import numpy as np
import imutils

image_data_path = r"ps5-images"
image_save_path = r"ps5-1"

# * Dilation and Erosion
def blobs(image, image_name, save=False, show=False):
    params = {"e": (3, 3), "d": (3, 3), "ei": 1, "di": 1}
    # morph(image)
    ekernel = np.ones(params["e"], dtype=np.uint8)
    dkernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, params["d"])
    if image_name == "wall1":  # Performing different operations on both the images
        image = cv.erode(image, ekernel, iterations=params["ei"])
        image = cv.dilate(image, dkernel, iterations=params["di"])
    else:
        image = cv.dilate(image, dkernel, iterations=params["di"])
        image = cv.erode(image, ekernel, iterations=params["ei"])

    if show:
        cv.namedWindow("Blob", cv.WINDOW_NORMAL)
        cv.imshow("Blob", image)
        cv.waitKey(0)
    if save:
        cv.imwrite(os.path.join(image_save_path, image_name + "-blobs.png"), image)

    return image


# * Contour Detection
def contours(image, image_name, save=False, show=False):
    # Added padding around image to avoid selecting the entire border as contour with a crack
    PAD_CONSTANT = 20
    image = cv.copyMakeBorder(
        image, PAD_CONSTANT, PAD_CONSTANT, PAD_CONSTANT, PAD_CONSTANT, borderType=cv.BORDER_CONSTANT, value=(255, 255, 255)
    )

    # Convert image to grayscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Convert image back to BGR to show contour colours
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    return_image = image.copy()

    thres = 0.98 if image_name == "wall1" else 0.99
    med = np.quantile(np.asarray([cv.arcLength(c, True) for c in contours]), thres)
    for i in range(len(contours)):
        peri = cv.arcLength(contours[i], True)  # Using perimeter as a parameter to threshold the cracks from the blobs
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))  # Random Color
        cv.drawContours(image, contours, i, color, 2)  # Draw coloured border around all blobs
        if not peri > med:
            cv.drawContours(return_image, contours, i, (255, 255, 255), thickness=-1)  # Draw over the small blobs

    # Resizing image back to original dimensions (remove padding)
    image_contours = image[PAD_CONSTANT : image.shape[0] - PAD_CONSTANT, PAD_CONSTANT : image.shape[1] - PAD_CONSTANT]
    return_image = return_image[PAD_CONSTANT : image.shape[0] - PAD_CONSTANT, PAD_CONSTANT : image.shape[1] - PAD_CONSTANT]

    if show:
        cv.namedWindow("Contours", cv.WINDOW_NORMAL)
        cv.imshow("Contours", image_contours)
        cv.waitKey(0)

    if save:
        cv.imwrite(os.path.join(image_save_path, image_name + "-contours.png"), image_contours)

    return return_image


# * Thinning/ Skeletonization
def skeleton(image, image_name, save=False, show=False):
    # Convert image to grayscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Invert image to use open and subtraction for thinning
    image = cv.bitwise_not(image)
    cross_kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    skeleton = np.zeros_like(image)

    while cv.countNonZero(image) != 0:
        eroded_image = cv.erode(image, cross_kernel)
        opened_image = cv.morphologyEx(eroded_image, cv.MORPH_OPEN, cross_kernel)
        subset = eroded_image - opened_image
        skeleton = cv.bitwise_or(subset, skeleton)
        image = eroded_image.copy()

    skeleton = cv.bitwise_not(skeleton)

    if show:
        cv.namedWindow("Skeleton", cv.WINDOW_NORMAL)
        cv.imshow("Skeleton", skeleton)
        cv.waitKey(0)

    if save:
        cv.imwrite(os.path.join(image_save_path, image_name + "-cracks.png"), skeleton)


if __name__ == "__main__":
    # * Reading the list of available images in the
    image_data = os.listdir(image_data_path)
    print("List of images = ", image_data)

    image_name = input("Enter the name of the image to process : ")

    image_path = os.path.join("ps5-images", image_name + ".png")
    image = cv.imread(image_path)
    if image is None:
        print("\nThe entered image could not be found!")
        exit(0)

    # * Processing the image (show = Show image in window, save = Save image in ps5-1 folder)
    # * Blob processing
    image_blobs = blobs(image, image_name, save=True, show=False)

    # * Contour Detection
    image_contours = contours(image_blobs, image_name, save=True, show=False)

    # * Skeletonization
    image_skeleton = skeleton(image_contours, image_name, save=True, show=False)

    print("\nProcessing Complete!")

    cv.destroyAllWindows()


# Trackbars to find best dilation and erosion combination
# def morph(image):
#     def change(val):
#         pass

#     cv.namedWindow("Blob", cv.WINDOW_NORMAL)
#     cv.createTrackbar("Dilation", "Blob", 1, 10, change)
#     cv.createTrackbar("Dilation iter", "Blob", 1, 10, change)
#     cv.createTrackbar("Erosion", "Blob", 1, 10, change)
#     cv.createTrackbar("Erosion iter", "Blob", 1, 10, change)
#     while True:
#         e = int(cv.getTrackbarPos("Erosion", "Blob"))
#         d = int(cv.getTrackbarPos("Dilation", "Blob"))
#         ei = int(cv.getTrackbarPos("Erosion iter", "Blob"))
#         di = int(cv.getTrackbarPos("Dilation iter", "Blob"))

#         # dkernel = np.ones((d, d), dtype=np.uint8)
#         # ekernel = np.ones((e, e), dtype=np.uint8)

#         dkernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (d, d))
#         ekernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (e, e))

#         image_dilated = cv.dilate(image, dkernel, iterations=di)
#         image_eroded = cv.erode(image_dilated, ekernel, iterations=ei)

#         # output = cv.bitwise_not(output)
#         cv.imshow("Blob", image_eroded)

#         if cv.waitKey(1) & 0xFF == ord("e"):
#             print("Erosion = ", e, " Dilation = ", d)
#             break
#     cv.destroyAllWindows()
