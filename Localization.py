import cv2
import numpy as np
import matplotlib.pyplot as plt


def plate_detection(image):
    """
    In this file, you need to define plate_detection function.
    To do:
        1. Localize the plates and crop the plates
        2. Adjust the cropped plate images
    Inputs:(One)
        1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
        type: Numpy array (imread by OpenCV package)
    Outputs:(One)
        1. plate_imgs: cropped and adjusted plate images
        type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
    Hints:
        1. You may need to define other functions, such as crop and adjust function
        2. You may need to define two ways for localizing plates(yellow or other colors)
    """

    mask = isolate_yellow_plate(image)

    mask = clear_noise(mask)

    coordinates = get_coordinates(mask)

    plate_images = get_plate(image, coordinates, 0)
    return plate_images


def isolate_yellow_plate(image):
    """
    creates a mask that isolates only the yellow objects (the plate as well)

    Parameters: frame from the video

    Returns: mask
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_min = np.array([10, 90, 90])
    color_max = np.array([45, 255, 255])

    mask = cv2.inRange(hsv_image, color_min, color_max)

    return mask


def clear_noise(image):
    """
    applies CLOSING operation for noise reduction

    Parameters: frame from the video

    Returns: denoised image
    """
    width = image.shape[1]

    kernel_size = int(0.02 * width)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    denoisedImage = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return denoisedImage


def get_coordinates(mask):
    """
    creates contours on a given mask based on yellow regions (regions of interest)
    finds the coordinates for every contour that resembles a rectangle
    checks for the coordinates validity

    Parameters: mask of a frame

    Returns: Array with all found rectangles' coordinates. A rectangle has 4 corners each, with (x,y) coordinates.
    The first corner is the BOTTOM-LEFT one, then the list continues in clockwise order.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coord = []

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect)
        corners = np.intp(corners)
        corners = sort_corners_from_bottom_left(corners)

        if check_coordinates_valid(corners):
            coord.append(corners)

    return coord

def sort_corners_from_bottom_left(corners):
    """
    checks if the corners start with BOTTOM LEFT corner
    if not, then they must start with TOP LEFT, and it changes the order to BOTTOM LEFT first

    Parameters: array with coordinates of the corners

    Return: BOTTOM LEFT starting point array
    """
    if np.abs(corners[0][0] - corners[1][0])>np.abs(corners[0][0] - corners[3][0]):
        bottom_left = corners[3].copy()
        corners[3] = corners[2]
        corners[2] = corners[1]
        corners[1] = corners[0]
        corners[0] = bottom_left

    return corners

def check_coordinates_valid(corners):
    """
    checks whether the coordinates of the corners are valid

    Parameters: array with coordinates of the corners

    Return: True or False depending on validity
    """

    side_length_1 = np.linalg.norm(corners[0] - corners[1])
    side_length_2 = np.linalg.norm(corners[1] - corners[2])

    # the bigger side_length will be width and the smaller will be height
    width = max(side_length_1, side_length_2)
    height = min(side_length_1, side_length_2)

    if width == 0 or height == 0:
        return False

    ratio = width / height

    # TODO: tweak the parameters and/or add more constraints
    if height * width < 200 or width < 50 or ratio < 2.5 or ratio > 5.5:
        return False
    return True


def rotate_tilted_image(image, corners):
    """
    rotate an image such that the plate is horizontally rotated
    Steps:  compute the angle by looking at the line through the top corners;
            compute a rotation matrix based on that angle
            rotate the image
            compute the updated corners by multiplying them with the rotation matrix

    Parameters: frame from the video, the 4 corners of the plate

    Returns: rotated image, updated corners
    """
    top_left = corners[1]
    top_right = corners[2]

    angle = np.degrees(np.arctan2((top_right[1]-top_left[1]), (top_right[0]-top_left[0])))

    height, width = image.shape[:2]

    center = (width//2, height//2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))


    corners = np.hstack([corners, np.ones((4, 1))])
    new_corners = np.dot(rotation_matrix, corners.T).T

    new_corners = np.intp(new_corners)

    return rotated_image, new_corners


def get_plate(image, coordinates, offset):
    """
    cuts off the plate from the given image
    takes an offset into account for a more covering cutting region

    Parameters: frame from the video, coordinates set, offset value

    Returns: cropped plate image
    """
    result_images = []

    for coord in coordinates:
        copy = image.copy()

        rotated_image, new_corners = rotate_tilted_image(copy, coord)

        x_left = min(new_corners[0][0], new_corners[1][0])
        x_right = max(new_corners[2][0], new_corners[3][0])

        y_up = min(new_corners[1][1], new_corners[2][1])
        y_down = max(new_corners[0][1], new_corners[3][1])

        width_0 = max(0, x_left - offset)
        width_1 = min(rotated_image.shape[1] - 1, x_right + offset)

        height_0 = max(0, y_up - offset)
        height_1 = min(image.shape[0] - 1, y_down + offset)

        result_images.append(rotated_image[height_0:height_1, width_0:width_1])

    return result_images


def localize(image):
    """
    combines all the steps to localize the plates

    Parameters: frame from the video

    Returns: localized plate image
    """
    mask = isolate_yellow_plate(image)

    mask = clear_noise(mask)

    coordinates = get_coordinates(mask)

    images = get_plate(image, coordinates, 0)

    return images





