# Import TensorFlow for ML operations
import tensorflow as tf
# Import Keras from TensorFlow for creating ML models
from tensorflow import keras
# Import BodyPart enum for body landmarks
from pose_estimation.pose_data_types import BodyPart
import pandas as pd                          # Import pandas for data manipulation


def get_center_point(landmarks, left_bodypart, right_bodypart, key_point_indices):
    """
    This function calculates the center point between two body parts.
    """
    left_index = key_point_indices.index(left_bodypart.value)
    right_index = key_point_indices.index(right_bodypart.value)

    print(key_point_indices, left_index, right_index)
    left = tf.gather(landmarks, left_index, axis=1)
    right = tf.gather(landmarks, right_index, axis=1)
    center = left * 0.5 + right * 0.5
    return center


def get_pose_size(landmarks, key_point_indices, torso_size_multiplier=2.5):
    """Calculates pose size.
    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """
    num_key_points = len(key_point_indices)
    # Shoulders center
    shoulders_center = get_center_point(
        landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER, key_point_indices)
    # Pose center (hips center)
    pose_center = get_center_point(
        landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP, key_point_indices)
    pose_center_expanded = tf.expand_dims(pose_center, axis=1)
    torso_size = tf.linalg.norm(shoulders_center - pose_center)
    pose_center_broadcasted = tf.broadcast_to(pose_center_expanded, [tf.size(
        landmarks) // (num_key_points * 2), num_key_points, 2])
    d = tf.gather(landmarks - pose_center_broadcasted,
                  0, axis=0, name="dist_to_pose_center")
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)
    return pose_size


def normalize_pose_landmarks(landmarks, key_point_indices):
    """Normalizes the landmarks translation by moving the pose center to (0,0) and
    scaling it to a constant pose size.
    """
    num_key_points = len(key_point_indices)
    # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(
        landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP, key_point_indices)
    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    pose_center = tf.broadcast_to(pose_center, [tf.size(
        landmarks) // (num_key_points * 2), num_key_points, 2])
    landmarks = landmarks - pose_center
    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(
        landmarks, key_point_indices, torso_size_multiplier=2.5)
    landmarks /= pose_size
    return landmarks


def landmarks_to_embedding(landmarks_and_scores, key_point_indices):
    """
    This function converts the landmarks into a pose embedding.
    """
    num_key_points = len(key_point_indices)
    # Reshape the input 1D tensor to a shape of a 2D tensor with [this.numKeyPoints, 3] (this.numKeyPoints landmarks with 3 items each)
    # The input shape is [num_key_points * 3] because the pose landmarks input data has this.numKeyPoints points, and each point has 3 items (x, y, and score).
    # This reshaping step makes it easier to work with the landmarks data and perform further processing,
    # like removing the confidence scores, normalizing the landmarks, and flattening them into an embedding.
    reshaped_inputs = keras.layers.Reshape(
        (num_key_points, 3))(landmarks_and_scores)

    # Normalize the 2D landmarks.
    # With reshaped_inputs[:, :, :2] we slice the tensor to remove the score, keeping only the x and y coordinates.
    # The score is removed because we want to analyze the pose in a 2D space (x and y coordinates only).
    landmarks = normalize_pose_landmarks(
        reshaped_inputs[:, :, :2], key_point_indices)

    # Flatten the normalized landmarks into a vector to create a pose embedding.
    embedding = keras.layers.Flatten()(landmarks)

    # Return the pose embedding.
    return embedding


def load_pose_landmarks(csv_path):
    """Loads a CSV created by MoveNetPreprocessor.
    Returns:
      X: Detected landmark coordinates and scores of shape (N, 17 * 3)
      y: Ground truth labels of shape (N, label_count)
      classes: The list of all class names found in the dataset
      dataframe: The CSV loaded as a Pandas dataframe features (X) and ground
        truth labels (y) to use later to train a pose classification model.
    """
    # Load the CSV file into a dataframe.
    dataframe = pd.read_csv(csv_path)
    df_to_process = dataframe.copy()

    # Drop the 'file_name' column as it's not needed for training.
    df_to_process.drop(columns=['file_name'], inplace=True)

    # Extract the unique class names.
    classes = df_to_process.pop('class_name').unique()

    # Extract the class numbers as labels.
    y = df_to_process.pop('class_no')

    # Convert the remaining columns into features for training.
    X = df_to_process.astype('float64')

    # Convert the labels into categorical format for training.
    y = keras.utils.to_categorical(y)

    # Return the features, labels, class names, and the original dataframe.
    return X, y, classes, dataframe
