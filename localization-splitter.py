import pandas as pd
import cv2
import numpy as np
import os

def split_on_category(category, training_percentage):
    """
    splits the category into training and test set

    Parameters:
        category (int): category number
        training_percentage (int): percentage of training set
    Returns:
        Training and test set
        type: the 2 objects are Dictionaries containing the plates
        and their info (formatted the same as in the csv)
    """

    ground_truth = pd.read_csv('dataset/groundTruth.csv')
    category_entries = ground_truth[ground_truth.Category == category]

    training_size = int((training_percentage/100) * len(category_entries))
    training_dictionary = category_entries[0:training_size]
    testing_dictionary = category_entries[training_size:]

    return training_dictionary, testing_dictionary

def clear_directory_contents(directory_path):
    """Removes all files from the specified directory but keeps the directory itself."""
    if os.path.exists(directory_path):
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):  # Remove files
                os.unlink(file_path)

# This file splits the dataset into train/test, by extracting 1 frame per license plate.
# It outputs the individual frames into the respective directory
if __name__ == '__main__':
    training_file_path = "localization-split/localization-training-set/"
    testing_file_path = "localization-split/localization-testing-set/"

    #split the data by category
    training_path = [training_file_path+"Category 1/", training_file_path+"Category 2/", training_file_path+"Category 3/", training_file_path+"Category 4/"]
    testing_path = [testing_file_path+"Category 1/", testing_file_path+"Category 2/", testing_file_path+"Category 3/", testing_file_path+"Category 4/"]

    # clean directories from previous files
    for i in range (4):
        clear_directory_contents(training_path[i])
        clear_directory_contents(testing_path[i])

    # choose split size (testing percentage is inferred with 100 - training_percentage)
    training_percentage = 70

    # dictionary with following information:
    # keys: frame number (int)
    # values: array with info about the frame [String:train or test, String:name of plate]
    frames_with_labels = {}

    # loop over all 4 categories
    for i in range(4):
        train_set, test_set = split_on_category(i+1, training_percentage)

        # pick the mean between first and last frame for each license plate
        for plate_info in train_set.values:
            visible_frame_number = int((plate_info[5]+plate_info[6])/2)
            plate_name = plate_info[2]

            # if there are 2 plates in 1 frame, we duplicate the image
            if visible_frame_number in frames_with_labels:
                frames_with_labels[visible_frame_number+1] = ["Train", plate_name, i + 1]
            else:
                frames_with_labels[visible_frame_number] = ["Train", plate_name, i+1]

        # do the same thing as before, but for the test set
        for plate_info in test_set.values:
            visible_frame_number = int((plate_info[5]+plate_info[6])/2)
            plate_name = plate_info[2]
            if visible_frame_number in frames_with_labels:
                frames_with_labels[visible_frame_number+1] = ["Test", plate_name, i + 1]
            else:
                frames_with_labels[visible_frame_number] = ["Test", plate_name, i+1]

    # now that we have all the relevant frames,
    # start processing the video
    video_file_path = 'dataset/trainingvideo.avi'
    cap = cv2.VideoCapture(video_file_path)

    if not cap.isOpened():
        print('Error, cannot open video file')

    # go over each frame
    frame_idx = 0

    while cap.isOpened():
      ret, frame = cap.read()
      # exit if there are no frames left
      if not ret:
          break

      # check if we hit a frame we're interested in
      if frame_idx in frames_with_labels.keys():
        frame_info = frames_with_labels[frame_idx]
        save_dir = None
        if frame_info[0] == "Train":
            save_dir = training_path[frame_info[2]-1]
        else:
            save_dir = testing_path[frame_info[2]-1]

        # allow duplicate plates in the file (if a plate appears multiple times during a video)
        # see plates 63-HK-HD and 56-JTT-5 in category 1

        filename = frame_info[1] + ".png"
        count = 1
        while(os.path.exists(save_dir+filename)):
            filename = f"{frame_info[1]}_{count}.png"
            count += 1

        cv2.imwrite(save_dir + filename, frame)

      frame_idx += 1
