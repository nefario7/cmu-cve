# Improving images using image filter kernels
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# * Read input image function
def read_image(image_name):
    image_path = os.path.join("ps3-images", image_name + ".png")
    image = cv.imread(image_path)
    if image is None:
        print("\nThe entered image could not be found!")
        exit(0)
    return image


# * Save processed image function
def save_image(image, image_name):
    save_folder = r"ps3-1"
    save_path = os.path.join(save_folder, image_name + "-improved.png")
    cv.imwrite(save_path, image)
    print("Image saved at : ", save_path)


# * Unsharp Masking function
def unsharp_masking(img, kernel, sigma, weights):
    gaussian = cv.GaussianBlur(img, kernel, sigma)
    unsharp_image = cv.addWeighted(img, weights[0], gaussian, weights[1], 0)
    return unsharp_image


# * Sharpening using a standard sharpening kernel
def sharpen_with_kernel(img):
    sk = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv.filter2D(img, ddepth=-1, kernel=sk)
    return sharpened_image


if __name__ == "__main__":
    """
    Use CLI to enter the image name to read and process
    """

    print("List of images [golf, pcb, rainbow, pots]")
    chosen_image = input("Choose which image to save with processing : ")

    if chosen_image == "pots":
        # *Processing for "pots.png" image ->
        # Sharpening + Unsharp Masking (9, 9) + Unsharp Masking (3, 3)
        pots_image = read_image("pots")
        pots_sharpened = sharpen_with_kernel(pots_image)
        pots_sharpened = sharpen_with_kernel(pots_sharpened)
        pots_sharpened = unsharp_masking(pots_sharpened, (9, 9), 7, [2, -1])
        pots_sharpened = unsharp_masking(pots_sharpened, (3, 3), 2, [2, -1])

        save_image(pots_sharpened, "pots")

    elif chosen_image == "golf":
        # *Processing for "golf.png" image ->
        # Median Filter + Sharpening + Unsharp Masking
        golf_image = read_image("golf")
        golf_median = cv.medianBlur(golf_image, 3)
        golf_sharpened = sharpen_with_kernel(golf_median)
        golf_sharpened = unsharp_masking(golf_sharpened, (3, 3), 1, [2, -1])
        save_image(golf_sharpened, "golf")

    elif chosen_image == "pcb":
        # *Processing for "pcb.png" image ->
        # Median Filter + Sharpening + Unsharp Masking
        pcb_image = read_image("pcb")
        pcb_median = cv.medianBlur(pcb_image, 3)
        pcb_sharpened = sharpen_with_kernel(pcb_median)
        pcb_sharpened = unsharp_masking(pcb_sharpened, (5, 5), 2, [1.5, -0.5])
        save_image(pcb_sharpened, "pcb")

    elif chosen_image == "rainbow":
        # *Processing for "rainbow.png" image ->
        # Bilateral Filter + Unsharp Masking
        rainbow_image = read_image("rainbow")
        rainbow_bilateral = cv.bilateralFilter(rainbow_image, 10, 75, 75)
        rainbow_sharpened = unsharp_masking(rainbow_bilateral, (9, 9), 3, [2, -1])
        save_image(rainbow_sharpened, "rainbow")
    else:
        print("Please enter a valid image to process!")
