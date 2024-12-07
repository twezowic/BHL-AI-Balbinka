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

    for folder, images in folder_data.items():
        if len(images) < 40:
            continue
        flag = True
        for image_path in images:
            if read_jpeg_metadata(image_path):
                flag = False
                break
        if flag:
            result.append(folder)
    return result

# TODO spłaszczyć foldery i chyba usunąć Active Region numbers
