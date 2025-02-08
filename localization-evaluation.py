import numpy as np
import os
import cv2
import pandas as pd
import Localization


def get_images_from_dir(path):
    """
    returns a list of all images from a directory

    Parameters:
        path -- the path of the directory
    Returns:
        array of tuples (image, file_name)
    """
    if not os.path.exists(path) or not os.path.isdir(path):
        print("Invalid directory path")
        return []

    images = []
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        images.append((cv2.imread(file_path), file_name.rstrip(".png")))

    return images


def intersection_over_union(rect1, rect2):
    """
    returns the intersection over union, calculated using masks

    Parameters:
        rect1, rect2 -- both contain the coordinates of their 4 corners (must be labeled in the same manner)
    Returns:
        intersection over union -- float between 0 and 1
    """
    # we use 640x480 masks because input images are always the same size
    rect1_mask = np.zeros((480, 640), dtype=np.uint8)
    rect2_mask = np.zeros((480, 640), dtype=np.uint8)

    # fill the masks according to rect1 and rect2
    cv2.drawContours(rect1_mask, [np.array(rect1)], 0, (255, 255, 255), thickness=cv2.FILLED)
    cv2.drawContours(rect2_mask, [np.array(rect2)], 0, (255, 255, 255), thickness=cv2.FILLED)

    intersection = cv2.bitwise_and(rect1_mask, rect2_mask)
    union = cv2.bitwise_or(rect1_mask, rect2_mask)

    # result
    iou = np.count_nonzero(intersection) / np.count_nonzero(union)
    return iou


def get_output_rectangle(image):
    """
    Uses the methods from Localization.py to calculate
    the coordinates of the license plate rectangle

    Parameters:
        image -- image containing the license plate
    Returns:
        coordinates of the 4 corners of rectangle as a 2d array
    """

    mask = Localization.isolate_yellow_plate(image)

    mask = Localization.clear_noise(mask)

    coordinates = Localization.get_coordinates(mask)

    if not coordinates:
        return []

    return list(coordinates[0])


def get_ground_truth_rectangle(plate_info):
    """
    extracts and formats the rectangle from the ground truth

    Parameters:
        plate_info -- this is just a row from the ground truth
    Outputs:
        list with coordinates of the rectangle's 4 corners ordered correctly
    """
    rectangle = [[plate_info.x1.values[0], plate_info.y1.values[0]], [plate_info.x2.values[0], plate_info.y2.values[0]],
                 [plate_info.x3.values[0], plate_info.y3.values[0]], [plate_info.x4.values[0], plate_info.y4.values[0]]]

    rectangle = format_rectangle(rectangle)
    return rectangle


def format_rectangle(rectangle):
    """
    orders the list with corners of a rectangle such that
    it's ordering corresponds to cv2's rotated rectangle labeling

    Parameters:
        rectangle -- list with coordinates of the rectangle's 4 corners
    Returns:
        list with coordinates of the rectangle's 4 corners ordered correctly
    """

    ordered_rectangle = []

    # the corner with lowest x-value comes first
    min_x_idx = 0
    for idx in range(len(rectangle)):
        if rectangle[idx][0] < rectangle[min_x_idx][0]:
            min_x_idx = idx

    # now label the corners in a clockwise manner
    # (because gt was labeled clockwise, we can just linearly increment starting from min_x_idx)
    for i in range(4):
        ordered_rectangle.append(rectangle[(i + min_x_idx) % 4])

    return ordered_rectangle


if __name__ == '__main__':
    ground_truth_path = "localization-split/localizationGroundTruth.csv"
    ground_truth = pd.read_csv(ground_truth_path)

    train_set_path = "localization-split/localization-training-set"
    test_set_path = "localization-split/localization-testing-set"

    # train and test set dictionaries,
    # with categories as keys, (images, file_name) as values
    train_set = {1: [], 2: [], 3: [], 4: []}
    test_set = {1: [], 2: [], 3: [], 4: []}

    # size of training set and test set
    train_set_size = 0
    test_set_size = 0

    for i in range(4):
        training_images = get_images_from_dir(train_set_path + f"/Category {i + 1}")
        testing_images = get_images_from_dir(test_set_path + f"/Category {i + 1}")

        train_set[i + 1] = training_images
        test_set[i + 1] = testing_images

        train_set_size += len(training_images)
        test_set_size += len(testing_images)

    # strictness in deciding for what is a "correctly localized plate"
    min_iou_threshold = 0.65

    # accuracy per category
    train_accuracy = np.zeros(4)
    test_accuracy = np.zeros(4)

    # overall accuracy
    combined_train_accuracy = 0
    combined_test_accuracy = 0

    for i in range(4):
        for (img, file_name) in train_set[i + 1]:
            output_rectangle = get_output_rectangle(img)
            if not output_rectangle:
                output_rectangle = [[0, 0], [0, 0], [0, 0], [0, 0]]
                print(f"No rectangle was localized for: {file_name}, from training set in category {i + 1}")

            plate_info = ground_truth[ground_truth["file name"] == file_name]
            ground_truth_rectangle = get_ground_truth_rectangle(plate_info)

            iou = intersection_over_union(output_rectangle, ground_truth_rectangle)
            if iou > min_iou_threshold:
                train_accuracy[i] += 1 / len(train_set[i + 1])
                combined_train_accuracy += 1 / train_set_size

        for (img, file_name) in test_set[i + 1]:
            output_rectangle = get_output_rectangle(img)
            if not output_rectangle:
                output_rectangle = [[0, 0], [0, 0], [0, 0], [0, 0]]
                print(f"No rectangle was localized for: {file_name}, from testing set in category {i + 1}")

            plate_info = ground_truth[ground_truth["file name"] == file_name]
            ground_truth_rectangle = get_ground_truth_rectangle(plate_info)

            iou = intersection_over_union(output_rectangle, ground_truth_rectangle)
            if iou > min_iou_threshold:
                test_accuracy[i] += 1 / len(test_set[i + 1])
                combined_test_accuracy += 1 / test_set_size

    print("------------------------------ results ------------------------------")

    for i in range(4):
        print(f"Training accuracy for category {i + 1}: {train_accuracy[i] * 100}%")
        print(f"Test accuracy for category {i + 1}: {test_accuracy[i] * 100}%")

    print("------------------------------ overall ------------------------------")

    print(f"Training accuracy overall: {combined_train_accuracy * 100}%")
    print(f"Testing accuracy overall: {combined_test_accuracy * 100}%")
