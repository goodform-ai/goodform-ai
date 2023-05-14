import tensorflow as tf
from tensorflow import keras
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflowjs as tfjs
from pose_estimation.pose_data_types import BodyPart

# Define functions to convert the pose landmarks to a pose embedding (a.k.a. feature vector) for pose classification


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


csvs_out_train_path = 'train_data.csv'
csvs_out_test_path = 'test_data.csv'


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


# Load the train data
X, y, class_names, _ = load_pose_landmarks(csvs_out_train_path)

# Split training data (X, y) into (X_train, y_train) and (X_val, y_val)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

# Load the test data
X_test, y_test, _, df_test = load_pose_landmarks(csvs_out_test_path)

# Define the model


def create_and_train_model(num_key_points=17):
    # base_key_points = [BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP,
    #                    BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER]

    # exercise_key_points = {
    #     "squat": base_key_points + [BodyPart.LEFT_KNEE, BodyPart.RIGHT_KNEE, BodyPart.LEFT_ANKLE, BodyPart.RIGHT_ANKLE],
    #     "push_up": base_key_points + [...],
    # }

    # key_points = exercise_key_points[exercise_name]
    # num_key_points = 17
    input_size = num_key_points * 3

    inputs = tf.keras.Input(shape=(input_size))
    embedding = landmarks_to_embedding(inputs, num_key_points)

    # Modify the number of nodes in the Dense layers based on the input size
    layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
    layer = keras.layers.Dropout(0.5)(layer)
    layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
    layer = keras.layers.Dropout(0.5)(layer)
    outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

    model = keras.Model(inputs, outputs)
    model.summary()

    # Compile and train model with training data

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Add a checkpoint callback to store the checkpoint that has the highest
    # validation accuracy.
    checkpoint_path = "weights.best.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 monitor='val_accuracy',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 mode='max')
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  patience=20)

    # Start training
    history = model.fit(X_train, y_train,
                        epochs=200,
                        batch_size=16,
                        validation_data=(X_val, y_val),
                        callbacks=[checkpoint, earlystopping])


    # Save the best model
    best_model = tf.keras.models.load_model(checkpoint_path)

    # Convert Keras model to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    tflite_model = converter.convert()

    tflite_model_filename = 'pose_classifier.tflite'

    # Save the TFLite model
    with open(tflite_model_filename, "wb") as f:
        f.write(tflite_model)

    input_format = 'tf_lite'
    output_format = 'tfjs_graph_model'
    output_dir = 'tfjsModel'

    tfjs.convert(input_format=input_format,
                output_format=output_format,
                output_node_names=None,  # This parameter is not needed for tf_lite input_format
                saved_model_tags=None,  # This parameter is not needed for tf_lite input_format
                quantization_dtype_map=None,  # This parameter is not needed for tf_lite input_format
                skip_op_check=False,
                strip_debug_ops=True,
                weight_shard_size_bytes=1024 * 1024 * 4,
                input_saved_model_dir=None,  # This parameter is not needed for tf_lite input_format
                # This parameter is not needed for tf_lite input_format
                input_saved_model_signature_key=None,
                input_tf_frozen_model=None,  # This parameter is not needed for tf_lite input_format
                input_tf_lite=tflite_model_filename,
                input_tf_hub=None,  # This parameter is not needed for tf_lite input_format
                output_dir=output_dir)
