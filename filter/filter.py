import os
import numpy as np
import cv2


def classify_image(image_path, red_orange_threshold=50):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([11, 50, 50])
    upper_orange = np.array([25, 255, 255])

    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    orange_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

    combined_mask = red_mask1 | red_mask2 | orange_mask

    red_orange_pixels = np.sum(combined_mask > 0)
    total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
    percentage = (red_orange_pixels / total_pixels) * 100

    return "useful" if percentage >= red_orange_threshold else "not useful"


root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "example_data")
useful = 0
useless = 0
for file_name in os.listdir(root):
    file_path = os.path.join(root, file_name)
    result = classify_image(file_path)
    if result == "useful":
        useful += 1
    else:
        useless += 1
    print(f"The image '{file_name}' is classified as: {result}")
print(f"{useful} images are useful")
print(f"{useless} images are not useful")
print(f"{useful+useless} images summary")
