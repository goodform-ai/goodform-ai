import csv
import cv2
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
# comes from data.py in https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/raspberry_pi
from pose_estimation.pose_data_types import BodyPart, Person, Point, KeyPoint
# comes from ml folder in https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/raspberry_pi
from pose_estimation.movenet_pose_estimation import Movenet
from pose_estimation.pose_visualization import visualize, keep_aspect_ratio_resizer

movenet = Movenet('models/movenet_thunder.tflite')


def detect(input_tensor, inference_count=3, key_point_indices=None):
    """Runs detection on an input image.

    # Define function to run pose estimation using MoveNet Thunder.
    # You'll apply MoveNet's cropping algorithm and run inference multiple times on
    # the input image to improve pose estimation accuracy.

    Args:
      input_tensor: A [height, width, 3] Tensor of type tf.float32.
        Note that height and width can be anything since the image will be
        immediately resized according to the needs of the model within this
        function.
      inference_count: Number of times the model should run repeatly on the
        same input image to improve detection accuracy.

    Returns:
      A Person entity detected by the MoveNet.SinglePose.
    """
    image_height, image_width, channel = input_tensor.shape

    # Detect pose using the full input image
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)

    # Repeatedly using previous detection result to identify the region of
    # interest and only croping that region to improve detection accuracy
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor.numpy(), reset_crop_region=False)

    # zero out keypoints
    # if key_point_indices is not None:
    #     all_keypoints = person.keypoints.copy()
    #     for i in range(len(all_keypoints)):
    #         if i not in key_point_indices:
    #             all_keypoints[i] = KeyPoint(
    #                 all_keypoints[i].body_part, Point(0, 0), float(0))
    #     person = Person(all_keypoints, person.bounding_box,
    #                     person.score, person.id)
    return person

# Functions to visualize the pose estimation results.


def draw_prediction_on_image(
        image, person, crop_region=None, close_figure=True,
        keep_input_size=False):
    """Draws the keypoint predictions on image.

    Args:
      image: An numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
      person: A person entity returned from the MoveNet.SinglePose model.
      close_figure: Whether to close the plt figure after the function returns.
      keep_input_size: Whether to keep the size of the input image.

    Returns:
      An numpy array with shape [out_height, out_width, channel] representing the
      image overlaid with keypoint predictions.
    """
    # Draw the detection result on top of the image.
    image_np = visualize(
        image, [person])

    # DEBUG TO SHOW IMAGES WHILE BEING GENERATED
    # img = cv2.resize(image, (1280, 800))
    # cv2.imshow("Visualized Image", img)  # Display the image in a window
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()  # Close the window and clean up

    # Plot the image with detection results.
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    im = ax.imshow(image_np)

    if close_figure:
        plt.close(fig)

    if not keep_input_size:
        image_np = keep_aspect_ratio_resizer(
            image_np, (512, 512))

    return image_np


def example_test():
    exercise_key_points = {
        "squat": [BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER, BodyPart.LEFT_KNEE, BodyPart.RIGHT_KNEE, BodyPart.LEFT_ANKLE, BodyPart.RIGHT_ANKLE],
    }

    key_points = exercise_key_points["squat"]
    key_point_indices = [kp.value for kp in key_points]

    image = tf.io.read_file('./image_test.jpeg')
    image = tf.io.decode_jpeg(image)
    person = detect(image, 3, key_point_indices)
    _ = draw_prediction_on_image(image.numpy(), person, crop_region=None,
                                 close_figure=False, keep_input_size=True)
    print(key_point_indices, key_points)


# Code to load the images, detect pose landmarks and save them into a CSV file
class MoveNetPreprocessor(object):
    """Helper class to preprocess pose sample images for classification."""

    def __init__(self,
                 images_in_folder,
                 images_out_folder,
                 csvs_out_path,
                 key_point_indices=None):
        """Creates a preprocessor to detection pose from images and save as CSV.

        Args:
          images_in_folder: Path to the folder with the input images. It should
            follow this structure:
            yoga_poses
            |__ downdog
                |______ 00000128.jpg
                |______ 00000181.bmp
                |______ ...
            |__ goddess
                |______ 00000243.jpg
                |______ 00000306.jpg
                |______ ...
            ...
          images_out_folder: Path to write the images overlay with detected
            landmarks. These images are useful when you need to debug accuracy
            issues.
          csvs_out_path: Path to write the CSV containing the detected landmark
            coordinates and label of each image that can be used to train a pose
            classification model.
        """
        print("MoveNetPreprocessor class initialized.")
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_path = csvs_out_path
        self.key_point_indices = key_point_indices
        self._messages = []
        self.inference_count = 3

        # Create a temp dir to store the pose CSVs per class
        self._csvs_out_folder_per_class = tempfile.mkdtemp()

        # Get list of pose classes and print image statistics
        self._pose_class_names = sorted(
            [n for n in os.listdir(self._images_in_folder)
             if not n.startswith('.')]
        )

    def process(self, per_pose_class_limit=None, detection_threshold=0.1):
        """Preprocesses images in the given folder.
        Args:
          per_pose_class_limit: Number of images to load. As preprocessing usually
            takes time, this parameter can be specified to make the reduce of the
            dataset for testing.
          detection_threshold: Only keep images with all landmark confidence score
            above this threshold.
        """
        # Loop through the classes and preprocess its images
        for pose_class_name in self._pose_class_names:
            print('Preprocessing', pose_class_name, file=sys.stderr)

            # Paths for the pose class.
            images_in_folder = os.path.join(
                self._images_in_folder, pose_class_name)
            images_out_folder = os.path.join(
                self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder_per_class,
                                        pose_class_name + '.csv')
            if not os.path.exists(images_out_folder):
                os.makedirs(images_out_folder)

            # Detect landmarks in each image and write it to a CSV file
            with open(csv_out_path, 'w') as csv_out_file:
                csv_out_writer = csv.writer(csv_out_file,
                                            delimiter=',',
                                            quoting=csv.QUOTE_MINIMAL)
                # Get list of images
                image_names = sorted(
                    [n for n in os.listdir(images_in_folder) if not n.startswith('.')])
                if per_pose_class_limit is not None:
                    image_names = image_names[:per_pose_class_limit]

                valid_image_count = 0

                # Detect pose landmarks from each image
                for image_name in tqdm.tqdm(image_names):
                    image_path = os.path.join(images_in_folder, image_name)

                    try:
                        image = tf.io.read_file(image_path)
                        image = tf.io.decode_jpeg(image)
                    except:
                        self._messages.append(
                            'Skipped ' + image_path + '. Invalid image.')
                        continue
                    else:
                        image = tf.io.read_file(image_path)
                        image = tf.io.decode_jpeg(image)
                        image_height, image_width, channel = image.shape

                    # Skip images that isn't RGB because Movenet requires RGB images
                    if channel != 3:
                        self._messages.append('Skipped ' + image_path +
                                              '. Image isn\'t in RGB format.')
                        continue

                    person = detect(image, self.inference_count,
                                    self.key_point_indices)

                    # Remove keypoints not in self.key_point_indices to display the keypoints we are focusing on for training data
                    # Get only the keypoints in self.key_point_indices
                    filtered_keypoints = [person.keypoints[i] for i in self.key_point_indices]
                    # Create a new Person object with these keypoints
                    person = Person(filtered_keypoints, person.bounding_box, person.score, person.id)

                    # Save landmarks if all landmarks were detected
                    min_landmark_score = min(
                        [keypoint.score for keypoint in person.keypoints])
                    should_keep_image = min_landmark_score >= detection_threshold
                    if not should_keep_image:
                        self._messages.append('Skipped ' + image_path +
                                              '. No pose was confidentlly detected.')
                        continue

                    valid_image_count += 1

                    # Draw the prediction result on top of the image for debugging later
                    output_overlay = draw_prediction_on_image(
                        image.numpy().astype(np.uint8), person,
                        close_figure=True, keep_input_size=True)

                    # Write detection result into an image file
                    output_frame = cv2.cvtColor(
                        output_overlay, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(
                        images_out_folder, image_name), output_frame)

                    # # Add missing keypoints back for csv to maintain 17 points
                    # if self.key_point_indices is not None:
                    #     # Create a full set of keypoints initialized to (0,0,0)
                    #     all_keypoints = [KeyPoint(body_part.value, Point(0, 0), 0.0) for body_part in BodyPart]
                        
                    #     # For each keypoint that was detected, replace the corresponding keypoint in all_keypoints
                    #     for keypoint in person.keypoints:
                    #         all_keypoints[keypoint.body_part.value] = keypoint

                    #     # Now all_keypoints contains all keypoints, either detected or default (0,0,0)
                    #     person = Person(all_keypoints, person.bounding_box, person.score, person.id)

                    # Get landmarks and scale it to the same size as the input image
                    pose_landmarks = np.array(
                        [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                         for keypoint in person.keypoints],
                        dtype=np.float32)

                    # Write the landmark coordinates to its per-class CSV file
                    coordinates = pose_landmarks.flatten().astype(np.str).tolist()
                    csv_out_writer.writerow([image_name] + coordinates)

                if not valid_image_count:
                    raise RuntimeError(
                        'No valid images found for the "{}" class.'
                        .format(pose_class_name))

        # Print the error message collected during preprocessing.
        print('\n'.join(self._messages))

        # Combine all per-class CSVs into a single output file
        all_landmarks_df = self._all_landmarks_as_dataframe()
        all_landmarks_df.to_csv(self._csvs_out_path, index=False)

    def class_names(self):
        """List of classes found in the training dataset."""
        return self._pose_class_names

    def _all_landmarks_as_dataframe(self):
        """Merge all per-class CSVs into a single dataframe."""
        total_df = None
        for class_index, class_name in enumerate(self._pose_class_names):
            csv_out_path = os.path.join(self._csvs_out_folder_per_class,
                                        class_name + '.csv')
            per_class_df = pd.read_csv(csv_out_path, header=None)

            # Add the labels
            per_class_df['class_no'] = [class_index]*len(per_class_df)
            per_class_df['class_name'] = [class_name]*len(per_class_df)

            # Append the folder name to the filename column (first column)
            per_class_df[per_class_df.columns[0]] = (os.path.join(class_name, '')
                                                     + per_class_df[per_class_df.columns[0]].astype(str))

            if total_df is None:
                # For the first class, assign its data to the total dataframe
                total_df = per_class_df
            else:
                # Concatenate each class's data into the total dataframe
                total_df = pd.concat([total_df, per_class_df], axis=0)

        list_name = [[bodypart.name + '_x', bodypart.name + '_y', bodypart.name + '_score'] for bodypart in BodyPart if bodypart.value in self.key_point_indices]
        header_name = []
        for columns_name in list_name:
            header_name += columns_name
        header_name = ['file_name'] + header_name
        header_map = {total_df.columns[i]: header_name[i]
                        for i in range(len(header_name))}

        total_df.rename(header_map, axis=1, inplace=True)

        return total_df
