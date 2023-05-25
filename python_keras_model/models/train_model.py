
from tensorflow import keras
from sklearn.model_selection import train_test_split
from models.pose_normalization import landmarks_to_embedding, load_pose_landmarks
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import datetime

def create_and_train_model(csvs_out_train_path, csvs_out_test_path, key_point_indices, test_size=0.15):
    """Create and train the model."""

    # Load training data
    X, y, class_names, _ = load_pose_landmarks(csvs_out_train_path)

    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

    # Load test data
    X_test, y_test, _, df_test = load_pose_landmarks(csvs_out_test_path)

    # Define the input size as the number of key points multiplied by 3
    # (x, y coordinates and a confidence score for each key point)
    input_size = len(key_point_indices) * 3

    # Define the input layer
    inputs = tf.keras.Input(shape=(input_size))

    # Use the landmarks_to_embedding function to process input data
    embedding = landmarks_to_embedding(inputs, key_point_indices=key_point_indices)

    # Define the neural network layers
    # The number of nodes in the Dense layers are based on the input size
    layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
    layer = keras.layers.Dropout(0.5)(layer)
    layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
    layer = keras.layers.Dropout(0.5)(layer)

    # The output layer has as many nodes as there are classes
    outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

    # Define the model
    model = keras.Model(inputs, outputs)
    model.summary()

    # Compile the model with the optimizer, loss function and metric
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Create a checkpoint callback to save the best model (highest validation accuracy)
    checkpoint_path = "weights.best.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 monitor='val_accuracy',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 mode='max')

    # Early stopping callback to stop training when validation accuracy does not improve for a certain number of epochs
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)

    # TensorBoard callback for visualizing training progress
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model
    history = model.fit(X_train, y_train,
                        epochs=200,
                        batch_size=16,
                        validation_data=(X_val, y_val),
                        callbacks=[checkpoint, earlystopping, tensorboard_callback])

    # Load the best model (highest validation accuracy) from the checkpoint
    best_model = tf.keras.models.load_model(checkpoint_path)

    return best_model, history
