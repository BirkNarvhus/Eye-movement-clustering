"""
Usage:
    python ./show_saved_video.py <numpy_file_path>
Description:
    Shows the saved video from the numpy file
    The numpy file should contain a list of numpy arrays
    Each numpy array should be a frame of the video
    The test_model.py is used to generate these files
"""

import cv2
import numpy as np
import sys
import os


def load_numpy_frame_data(file_path):
    """
    Load the numpy frame data from the file
    :param file_path:  the path to the numpy file
    :return:  the numpy frame data
    """
    return np.load(file_path)


def show_saved_video(file_path):
    """
    Show the saved video from the numpy file
    :param file_path:  the path to the numpy file
    """
    data = load_numpy_frame_data(file_path)
    print("Loaded video data of shape: ", data.shape, " from: ", file_path)
    for frame in data:
        cv2.imshow('Grayscale', frame)
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide the path to the numpy file containing the video frames")
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print("The file does not exist")
        sys.exit(1)
    show_saved_video(sys.argv[1])
