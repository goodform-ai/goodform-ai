
import tensorflow as tf
from tensorflow import keras
from pose_estimation.pose_data_types import BodyPart
import pandas as pd

def get_center_point(landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of the two given landmarks."""

    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5
    return center


def get_pose_size(landmarks, torso_size_multiplier=2.5, num_key_points=17):
    """Calculates pose size.

    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """
    # Shoulders center
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER,
                                        BodyPart.RIGHT_SHOULDER)

    # Pose center (hips center)
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                   BodyPart.RIGHT_HIP)
    pose_center_expanded = tf.expand_dims(pose_center, axis=1)

    # Torso size as the minimum body size
    torso_size = tf.linalg.norm(shoulders_center - pose_center)

    # Broadcast the pose center to the same size as the landmark vector to
    # perform subtraction
    pose_center_broadcasted = tf.broadcast_to(pose_center_expanded,
                                              [tf.size(landmarks) // (num_key_points * 2), num_key_points, 2])

    # Dist to pose center
    d = tf.gather(landmarks - pose_center_broadcasted, 0, axis=0,
                  name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)

    return pose_size


def normalize_pose_landmarks(landmarks, num_key_points=17):
    """Normalizes the landmarks translation by moving the pose center to (0,0) and
    scaling it to a constant pose size.
    """
    # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                   BodyPart.RIGHT_HIP)
    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    pose_center = tf.broadcast_to(pose_center,
                                  [tf.size(landmarks) // (num_key_points * 2), num_key_points, 2])
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks, 2.5, num_key_points)
    landmarks /= pose_size

    return landmarks


def landmarks_to_embedding(landmarks_and_scores, num_key_points=17):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(num_key_points, 3)
    reshaped_inputs = keras.layers.Reshape(
        (num_key_points, 3))(landmarks_and_scores)
    print(reshaped_inputs[:, :, :2])
    # Normalize landmarks 2D
    landmarks = normalize_pose_landmarks(
        reshaped_inputs[:, :, :2], num_key_points)

    # Flatten the normalized landmark coordinates into a vector
    embedding = keras.layers.Flatten()(landmarks)

    return embedding

# Load the preprocessed CSVs into TRAIN and TEST datasets.

def load_pose_landmarks(csv_path):
    """Loads a CSV created by MoveNetPreprocessor.

    Returns:
      X: Detected landmark coordinates and scores of shape (N, 17 * 3)
      y: Ground truth labels of shape (N, label_count)
      classes: The list of all class names found in the dataset
      dataframe: The CSV loaded as a Pandas dataframe features (X) and ground
        truth labels (y) to use later to train a pose classification model.
    """

    # Load the CSV file
    dataframe = pd.read_csv(csv_path)
    df_to_process = dataframe.copy()

    # Drop the file_name columns as you don't need it during training.
    df_to_process.drop(columns=['file_name'], inplace=True)

    # Extract the list of class names
    classes = df_to_process.pop('class_name').unique()

    # Extract the labels
    y = df_to_process.pop('class_no')

    # Convert the input features and labels into the correct format for training.
    X = df_to_process.astype('float64')
    y = keras.utils.to_categorical(y)

    return X, y, classes, dataframe
