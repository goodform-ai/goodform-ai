# Define function to Split images into training & test data

import os
import random
import shutil


def split_into_train_test(images_origin, images_dest, test_split):
    """Splits a directory of sorted images into training and test sets.

    Args:
      images_origin: Path to the directory with your images. This directory
        must include subdirectories for each of your labeled classes. For example:
        yoga_poses/
        |__ downdog/
            |______ 00000128.jpg
            |______ 00000181.jpg
            |______ ...
        |__ goddess/
            |______ 00000243.jpg
            |______ 00000306.jpg
            |______ ...
        ...
      images_dest: Path to a directory where you want the split dataset to be
        saved. The results looks like this:
        split_yoga_poses/
        |__ train/
            |__ downdog/
                |______ 00000128.jpg
                |______ ...
        |__ test/
            |__ downdog/
                |______ 00000181.jpg
                |______ ...
      test_split: Fraction of data to reserve for test (float between 0 and 1).
    """
    _, dirs, _ = next(os.walk(images_origin))

    TRAIN_DIR = os.path.join(images_dest, 'train')
    TEST_DIR = os.path.join(images_dest, 'test')
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    for dir in dirs:
        # Get all filenames for this dir, filtered by filetype
        filenames = os.listdir(os.path.join(images_origin, dir))
        filenames = [os.path.join(images_origin, dir, f) for f in filenames if (
            f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.bmp'))]
        # Shuffle the files, deterministically
        filenames.sort()
        random.seed(42)
        random.shuffle(filenames)
        # Divide them into train/test dirs
        os.makedirs(os.path.join(TEST_DIR, dir), exist_ok=True)
        os.makedirs(os.path.join(TRAIN_DIR, dir), exist_ok=True)
        test_count = int(len(filenames) * test_split)
        for i, file in enumerate(filenames):
            if i < test_count:
                destination = os.path.join(
                    TEST_DIR, dir, os.path.split(file)[1])
            else:
                destination = os.path.join(
                    TRAIN_DIR, dir, os.path.split(file)[1])
            shutil.copyfile(file, destination)
        print(
            f'Moved {test_count} of {len(filenames)} from class "{dir}" into test.')
    print(f'Your split dataset is in "{images_dest}"')


def split_init(path):
    # path: Creates a test and train folder, cuts every workout in {workout folder} and places them back into test and train

    # You can leave the rest alone:
    if not os.path.isdir(path):
        raise Exception("path is not a valid directory")

    dataset_out_parent = os.path.dirname(path)
    dataset_out_folder = 'split_' + os.path.basename(path)
    dataset_out = os.path.join(dataset_out_parent, dataset_out_folder)

    # Check if the 'split_{path}' directory and 'train' and 'test' folders exist
    train_folder_exists = os.path.isdir(os.path.join(dataset_out, 'train'))
    test_folder_exists = os.path.isdir(os.path.join(dataset_out, 'test'))

    if not train_folder_exists or not test_folder_exists:
        # If either the train or test folder doesn't exist, perform the split
        split_into_train_test(path, dataset_out, test_split=0.2)
        print(f'Dataset split into train and test folders.')
    else:
        print(f'Training and test data already exist.')

    IMAGES_ROOT = dataset_out
    return IMAGES_ROOT
