import cv2
import os
import pandas as pd
import numpy as np
import Localization
import Recognize
import matplotlib.pyplot as plt


def CaptureFrame_Process(file_path, sample_frequency, save_path):
    """
    In this file, you will define your own CaptureFrame_Process function. In this function,
    you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
    To do:
        1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
        2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
        3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
    Inputs:(three)
        1. file_path: video path
        2. sample_frequency: second
        3. save_path: final .csv file path
    Output: None
    """

    # read the frames and localize and recognize the plates
    plates = read_localize_recognize(file_path, sample_frequency)
    #group plates that are considered to be of the same car
    groups = group_similar_plates(plates,sample_frequency)
    #apply majority voting to identify the most "popular" license plate number for a car
    plate_numbers = majority_voting(groups)
    #do a Dutch plate validity check and place dashes
    result = []
    for number in plate_numbers:
        actual_plate = plate_validity_check(number[0])
        if(actual_plate != ''):
            result.append((actual_plate, number[1],number[2]))
    #write the ouput in the csv file
    output = open(save_path, "w")
    output.write("License plate,Frame no.,Timestamp(seconds)\n")
    for row in result:
        output.write(f"{row[0]},{row[1]},{row[2]}\n")
    output.close()
    pass


def read_localize_recognize(file_path, sample_frequency):
    """
    Goes through the input video frame by frame. Then localize and recognize the plates in the frame.
    If a plate is recognized, save the string, along with the frame number and the timestamp
    Input:
        file_path: video path
        sample_frequency: sampling frequency
    Output:
        result: list of "triples" containing the above-mentioned information
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print('Error, cannot open video file')
    ret, frame = cap.read()
    result = []
    frame_number = 1
    while ret:
        frame_number += 1
        frame = np.array(frame)
        plates = Localization.plate_detection(frame)
        if plates is None:
            ret, frame = cap.read()
            continue
        for plate in plates:
            if np.any(plate):
                letters = Recognize.segment_and_recognize(plate)
                if letters != '':
                    result.append((letters, frame_number, frame_number/sample_frequency))
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    return result

def group_similar_plates(plates, sample_frequency, frame_distance = 50, threshold = 3):
    """
    Group plates that MIGHT correspond to the same car. Average the frame number and the timestamp.
    Input:
        plates: list of "triplets" from previous method
        sample_frequency
        threshold: max number of differences between 2 plate strings
        frame_distance: max acceptable distance between frames
    Ouput:
        groups: list of groups, where a group has all strings of a single plate, average frame number
                and average timestamp
    """
    copy = []
    for plate in plates:
        copy.append(plate)

    groups = []
    while len(copy) > 0:
        first = copy[0]
        total_plates = 1
        total_frames = first[1]
        group = [first]
        copy.remove(first)
        for plate in copy:
            if plate[1] - first[1] <= frame_distance and difference_score(first[0], plate[0]) <= threshold:
                total_plates += 1
                total_frames += plate[1]
                group.append(plate)
        for plate in group:
            if plate in copy:
                copy.remove(plate)
        avg_frame = total_frames//total_plates
        groups.append((group, avg_frame, avg_frame/sample_frequency))
    return groups

def difference_score(plate1, plate2):
    """
    Computes the difference between 2 strings by comparing them letter by letter.
    Input:
        plate1, plate2: the 2 strings
    Ouput:
        score: difference score
    """
    score = 0
    for i in range(min(len(plate1), len(plate2))):
        if plate1[i] != plate2[i]:
            score += 1
    score += max(len(plate1), len(plate2)) - min(len(plate1), len(plate2))
    return score

def majority_voting(groups):
    """
    It takes every string that was recognized for a POSSIBLY single plate and builds a new string
    by taking for each of the 6 positions (Dutch plate format), the most present letter on that position.
    Input:
        groups: list of groups, where a group has all strings of a single plate, frame number and timestamp
    Ouput:
        result: an update "groups" list, with a single string associated to every plate
                (and thus with every single group)
    """
    result = []
    for group in groups:
        new_plate = ''
        for i in range(6):
            count = {}
            for plate in group[0]:
                if len(plate[0]) != 6:
                    continue
                letter = plate[0][i]
                if letter in count:
                    count[letter] += 1
                else:
                    count[letter] = 1
            if len(count) > 0:
                new_plate += max(count, key=count.get)
        result.append((new_plate, group[1], group[2]))
    return result


def plate_validity_check(plate_combination):
    plate = ""

    if (len(plate_combination) != 6):
        return plate

    validity_array = []
    for letter in plate_combination:
        if letter.isdigit():
            validity_array.append(False)
        elif letter.isalpha():
            validity_array.append(True)
        else:
            return ""

    # 3 letters/digits - 2 digits/letter - 1 letter/digit
    if np.array_equal(validity_array, [False, False, False, True, True, False]) \
            or np.array_equal(validity_array, [True, True, True, False, False, True]):
        plate = plate_combination[0] + plate_combination[1] + plate_combination[2] + "-" + plate_combination[3] + \
                plate_combination[4] + "-" + plate_combination[5]
    # 2 letter/digits - 3 digits/letters - 1 letter/digit
    elif np.array_equal(validity_array, [False, False, True, True, True, False]) \
             or np.array_equal(validity_array, [True, True, False, False, False, True]):
        plate = plate_combination[0] + plate_combination[1] + "-" + plate_combination[2] + plate_combination[3] + \
                plate_combination[4] + "-" + plate_combination[5]
    # 1 letter/digit - 3 digits/letters - 2 letters/digits
    elif np.array_equal(validity_array, [False, True, True, True, False, False]) \
             or np.array_equal(validity_array, [True, False, False, False, True, True]):
        plate = plate_combination[0] + "-" + plate_combination[1] + plate_combination[2] + plate_combination[3] + "-" + \
                plate_combination[4] + plate_combination[5]
    # 2 letter/digits - 2 digits/letters - 2 letters/digits
    elif np.array_equal(validity_array, [True, True, False, False, True, True]) \
            or np.array_equal(validity_array, [False, False, True, True, False, False]) \
            or np.array_equal(validity_array, [False, False, False, False, True, True]) \
            or np.array_equal(validity_array, [True, True, False, False, False, False]) \
            or np.array_equal(validity_array, [True, True, True, True, False, False]) \
            or np.array_equal(validity_array, [False, False, True, True, True, True]):
        plate = plate_combination[0] + plate_combination[1] + "-" + plate_combination[2] + plate_combination[3] + "-" + \
                plate_combination[4] + plate_combination[5]
    else:
        return ''
    return plate