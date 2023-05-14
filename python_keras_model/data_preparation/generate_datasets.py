import os
from pose_estimation.pose_estimation_preprocessing import MoveNetPreprocessor
from utils.directory_utils import update_to_keypoint_directory, update_to_pose_output_directory

def generate_set(dataset_root, train_folder='train', test_folder='test', key_point_indices=None):
    """
    dataset_root: The root folder containing the train and test datasets.
    train_folder: The folder name inside dataset_root containing the training images(default: 'train').
    test_folder: The folder name inside dataset_root containing the testing images(default: 'test').
    images_out_train_folder: The folder to store the processed training images with landmarks overlay(default: 'poses_images_out_train').
    images_out_test_folder: The folder to store the processed testing images with landmarks overlay(default: 'poses_images_out_test').
    csvs_out_train_path: The file path to store the CSV file containing the landmarks for the training images(default: 'train_data.csv').
    csvs_out_test_path: The file path to store the CSV file containing the landmarks for the testing images(default: 'test_data.csv').
    key_point_indices: Keypoints to focus on for the movenet processing detection
    """
    # Update the root directory
    updated_csvs_directory = update_to_keypoint_directory(dataset_root)
    # Create the full path for the CSV files
    csvs_out_train_path = os.path.join(updated_csvs_directory, 'train_data.csv')
    csvs_out_test_path = os.path.join(updated_csvs_directory, 'test_data.csv')

    # Update the root directory
    updated_images_out_directory = update_to_pose_output_directory(dataset_root)
    images_out_train_folder = os.path.join(updated_images_out_directory, "train")
    images_out_test_folder = os.path.join(updated_images_out_directory, "test")

    # Check if train CSV already exists
    if not os.path.exists(csvs_out_train_path):
        print("CSV not found, training train dataset")
        # Runs Training data through movenet and outputs a csv
        images_in_train_folder = os.path.join(dataset_root, train_folder)

        preprocessor = MoveNetPreprocessor(
            images_in_folder=images_in_train_folder,
            images_out_folder=images_out_train_folder,
            csvs_out_path=csvs_out_train_path,
            key_point_indices=key_point_indices
        )

        preprocessor.process(per_pose_class_limit=None)
    else:
        print("Train CSV already exists, skipping preprocessing for training data.")

    # Check if test CSV already exists
    if not os.path.exists(csvs_out_test_path):
        print("CSV not found, training test dataset")
        # Runs Test data through movenet and outputs a csv
        images_in_test_folder = os.path.join(dataset_root, test_folder)

        preprocessor = MoveNetPreprocessor(
            images_in_folder=images_in_test_folder,
            images_out_folder=images_out_test_folder,
            csvs_out_path=csvs_out_test_path,
            key_point_indices=key_point_indices
        )

        preprocessor.process(per_pose_class_limit=None)
    else:
        print("Test CSV already exists, skipping preprocessing for testing data.")

    return csvs_out_test_path, csvs_out_train_path
