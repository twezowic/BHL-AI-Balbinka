""" This module helps locate directories in the main project directory.
"""
import sys
import os
from pathlib import Path
import pandas as pd
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# path for the project directory
project_dir = Path(__file__).resolve().parents[3]


import os
from PIL import Image


def read_jpeg_metadata(file_path):
    image = Image.open(file_path)
    exif_data = image._getexif()
    return bool(exif_data)


def process_directory(base_path):
    folder_data = {}
    for root, dirs, files in os.walk(base_path):
        images = [
            os.path.join(root, file)
            for file in files
            if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg")
        ]
        folder_data[root] = images
    return folder_data


def filter_data(base_path):
    """
    Remove folders in which:
    - There are less than 40 images.
    - One of the images has "flagged" in the ImageDescription metadata (EXIF).
      It means raw data from the satellite is flagged as noisy, e.g. because of a moon eclipse or because of a recalibration.
    """
    folder_data = process_directory(base_path)
    result = []
    labels = pd.read_csv(base_path + '/meta_data.csv')
    for folder, images in folder_data.items():
        id = '_'.join(folder.split(os.sep)[-2:])
        if len(images) < 40 or id not in labels['id'].values:
            continue
        flag = True
        for image_path in images:
            if read_jpeg_metadata(image_path):
                flag = False
                break
        if flag:
            result.append(folder)
    return result


def get_data_dir() -> Path:
    """
    Gets raw directory path.
    """
    return project_dir / 'data'


def get_results_dir() -> Path:
    """
    Gets results directory path.
    """
    return project_dir / 'results'


def get_reports_dir() -> Path:
    """
    Gets reports directory path.
    """
    return project_dir / 'reports'


def get_references_dir() -> Path:
    """
    Gets references directory path.
    """
    return project_dir / 'references'


def main_load(dataset):
    training_path = dataset
    folders = filter_data(training_path)
    return folders


# print(main_load())