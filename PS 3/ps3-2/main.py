from utility import *
from detectors import *


if __name__ == "__main__":
    name = input("Enter image name to process [cheerios, professor, gear, circuit]: ")
    detector = int(input("Which detector to use? (1. Sobel Filter | 2. Canny Edge Detector) : "))

    image_original = read_image(name)
    image_gray = convert_to_grayscale(image_original)

    if detector == 1:
        # Sobel Filter to Detect Image
        blur_values = {"cheerios": [2, 3], "professor": [1, 3], "gear": [2, 5], "circuit": [1, 3]}
        image_blur = gaussian_blur(image_gray, blur_values[name][0], blur_values[name][1], average=True)
        image_sobel = sobel_filter(image_blur)
        save_image(image_sobel, name, "sobel")

    elif detector == 2:
        image_canny, params = canny_edge_detector(image_gray)
        save_image(image_canny, name, "canny")

    else:
        print("\nPlease select a detector from the provided list!")
