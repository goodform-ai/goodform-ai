
from tensorflow import keras
from sklearn.model_selection import train_test_split
from models.pose_normalization import landmarks_to_embedding, load_pose_landmarks
import tensorflow as tf

def create_and_train_model(csvs_out_train_path, csvs_out_test_path, test_size=0.15, num_key_points=17):
    """Create and train the model."""

    # Load the train data
    X, y, class_names, _ = load_pose_landmarks(csvs_out_train_path)

    # Split training data (X, y) into (X_train, y_train) and (X_val, y_val)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

    # Load the test data
    X_test, y_test, _, df_test = load_pose_landmarks(csvs_out_test_path)

    input_size = num_key_points * 3
    inputs = tf.keras.Input(shape=(input_size))
    embedding = landmarks_to_embedding(inputs, num_key_points)

    # Modify the number of nodes in the Dense layers based on the input size
    layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
    layer = keras.layers.Dropout(0.5)(layer)
    layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
    layer = keras.layers.Dropout(0.5)(layer)
    outputs = keras.layers.Dense(len(class_names), activation="sigmoid")(layer)

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

    return best_model, history
