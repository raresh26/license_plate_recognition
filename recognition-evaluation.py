import numpy as np
import os
import cv2
import pandas as pd
from matplotlib import pyplot as plt

import Recognize

def get_plates_from_directory(path):
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
        plate = cv2.imread(file_path)
        images.append((plate, file_name.rstrip(".png")))

    return images

def evaluate_recognition(plates):
    correct_plates = 0
    wrong_plates_result = {}

    for plate,file_name in plates:
        plate_images = []
        plate_images.append(plate)
        plate_texts = Recognize.segment_and_recognize(plate_images)
        ok = False

        for plate_text in plate_texts:
            if plate_text is not None and plate_text != '' and plate_text in file_name:
                correct_plates += 1
                ok = True
                break
        if not ok:
            wrong_plates_result[file_name] = plate_texts

    return correct_plates, wrong_plates_result

if __name__ == '__main__':
    train_path = "recognition-training-set"
    test_path = "recognition-testing-set"
    path = "recognition-split/"

    train_set_size = 0
    test_set_size = 0

    train_accuracy = np.zeros(4)
    test_accuracy = np.zeros(4)

    combined_train_accuracy = 0
    combined_test_accuracy = 0
    # training set
    for i in range(1,5):
        plates_path = path + train_path + f"/Category {i}"
        plates = get_plates_from_directory(plates_path)

        correct_plates, wrong_plates_result = evaluate_recognition(plates)

        train_set_size += len(plates)

        for file_name in wrong_plates_result.keys():
            print(f"Plate {file_name} in training set - Category {i} was wrongly recognized as {wrong_plates_result[file_name]}")

        train_accuracy[i-1] = correct_plates/len(plates)
        combined_train_accuracy += correct_plates
    # testing set
    for i in range(1, 5):

        plates_path = path + test_path + f"/Category {i}"
        plates = get_plates_from_directory(plates_path)

        correct_plates, wrong_plates_result = evaluate_recognition(plates)

        test_set_size += len(plates)

        for file_name in wrong_plates_result.keys():
            print(f"Plate {file_name} in testing set - Category {i} was wrongly recognized as {wrong_plates_result[file_name]}")

        test_accuracy[i - 1] = correct_plates / len(plates)
        combined_test_accuracy += correct_plates

    print("------------------------------ results ------------------------------")

    for i in range(4):
        print(f"Training accuracy for Category {i + 1}: {train_accuracy[i] * 100}%")
        print(f"Test accuracy for Category {i + 1}: {test_accuracy[i] * 100}%")

    print("------------------------------ overall ------------------------------")

    print(f"Training accuracy overall: {combined_train_accuracy/train_set_size * 100}%")
    print(f"Testing accuracy overall: {combined_test_accuracy / test_set_size * 100}%")

