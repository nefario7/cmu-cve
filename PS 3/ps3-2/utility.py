import os
import numpy as np
import cv2 as cv
import warnings
from tqdm import tqdm

# * UTILITY FUNCTIONS

# * Read input image
def read_image(image_name):
    image_path = os.path.join("./ps3-images", image_name + ".png")
    image = cv.imread(image_path)
    if image is None:
        print("\nThe entered image could not be found!")
        exit(0)
    return image


# * Save edge image function
def save_image(image, image_name, ext):
    save_folder = r"./ps3-2"
    save_path = os.path.join(save_folder, image_name + "-" + ext + ".png")
    cv.imwrite(save_path, image)
    print("Image saved at : ", save_path)


# * Convert image to grayscale
def convert_to_grayscale(image):
    # return (image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114) / 3.0
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


# * Convolution
def apply_convolution(image, kernel):
    # * Padding existing image based on kernel size
    vpad = kernel.shape[0] // 2  # Vertical Padding from no. of kernel rows
    hpad = kernel.shape[1] // 2  # Horizontal Padding from no. of kernel columns

    padded_image = np.zeros((image.shape[0] + 2 * vpad, image.shape[1] + 2 * hpad))
    padded_image[vpad:-vpad, hpad:-hpad] = image

    # * Convoling the image with kernel
    convolved_image = np.zeros(image.shape)
    for row in tqdm(range(image.shape[0]), desc="Applying Convolution"):
        for col in range(image.shape[1]):
            convolved_image[row, col] = np.sum(kernel * padded_image[row : (row + kernel.shape[0]), col : (col + kernel.shape[1])])

    return convolved_image


def gaussian_blur(input_image, sigma, kernel_size, average=False):
    def gammafunc(x, sigma):
        c = 1 / np.sqrt(2 * np.pi * sigma)
        p = (-0.5) * np.power(x / sigma, 2)
        return c * np.e ** p

    if kernel_size % 2 == 0:
        warnings.warn("Kernel size is not odd!")
    # assert kernel_size % 2 == 1, "Kernel Size should be odd"
    kernel1d = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    kernel1d = np.array([gammafunc(x, sigma) for x in kernel1d])  # Applying gamma function for x values
    kernel2d = np.outer(kernel1d.T, kernel1d.T)  # multiplying 2 1D vectors to get matrix
    kernel2d *= 1.0 / kernel2d.max()  # normalization

    # plt.imshow(kernel2d, interpolation='none', cmap='gray')
    # plt.title("Kernel")
    # plt.show()
    blurred_image = apply_convolution(input_image, kernel2d)
    if average:
        blurred_image = blurred_image / np.sum(kernel2d)
    return blurred_image
