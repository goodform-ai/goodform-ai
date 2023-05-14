import os

def update_to_keypoint_directory(dir_path):
    # get the parent directory
    parent_dir = os.path.dirname(dir_path)

    # get the last part of the directory
    last_part = os.path.basename(parent_dir)

    # construct the new directory
    new_dir = os.path.join(parent_dir, f'{last_part}_keypoint_data')

    return new_dir

def update_to_pose_output_directory(dir_path):
    # get the parent directory
    parent_dir = os.path.dirname(dir_path)

    # get the last part of the directory
    last_part = os.path.basename(parent_dir)

    # construct the new directory
    new_dir = os.path.join(parent_dir, f'{last_part}_keypoint_images')

    return new_dir
