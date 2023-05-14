import tensorflow as tf
import os
import subprocess

def convert_model(model, model_output_dir):
    """Convert a trained model to TFLite and TensorFlow.js formats."""
    # Define the paths for the output files
    tflite_output_path = os.path.join(model_output_dir, "model.tflite")
    tfjs_output_path = os.path.join(model_output_dir, "tfjs_model")
    keras_output_path = os.path.join(model_output_dir, "keras_model")

    # Save the model in Keras format
    model.save(keras_output_path)

    # Convert Keras model to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)

    # Convert the Keras model to TensorFlow.js format
    # Note that this requires tensorflowjs to be installed and accessible in the command line.
    subprocess.run(["tensorflowjs_converter", "--input_format=tf_saved_model", "--output_format=tfjs_graph_model", keras_output_path, tfjs_output_path], check=True)

    print(f"Model saved in Keras format and saved to {keras_output_path}.")
    print(f"Model converted to TFLite and saved to {tflite_output_path}.")
    print(f"Model converted to TensorFlow.js and saved to {tfjs_output_path}.")
