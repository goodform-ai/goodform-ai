from cv2 import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.lite as tflite

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

KEYPOINT_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head Keypoints
    (5, 6), (5, 7), (5, 11), (6, 8), (6, 12), (7, 9), (8, 10),  # Arm Keypoints
    (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Leg Keypoints
]


def load_tflite_model(tflite_model_path: str):
    interpreter = tflite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_tflite_model(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def process_pose_classifier_output(output_data, threshold=0.5):
    # Process the output data depending on your model's output format
    # For example, if it's a simple classification model, you may return the class index with the highest probability
    output_data = np.array(output_data)
    # Since it's a sigmoid output, we don't need argmax
    # Instead, we check if the output is greater than or equal to the threshold
    predicted_classes = (output_data >= threshold).astype(int)

    # Assuming you have a list of class names like this:
    class_names = ['bad-back', 'bad-knee-valgus', 'good']

    labels = [class_names[i] for i, is_present in enumerate(predicted_classes[0]) if is_present]

    print("Pose Classification Result:", labels, predicted_classes)


def get_color_by_score(score: float):
    """Calculates a shade of green depending on the passed score
    :param float score: A float between 0 and 1.
    :return: a tuple of 3 ints from 0 to 255 representing the RGB color
    """
    return 0, int(255 * score), 0


def draw_markers(keypoints_xys: np.ndarray, frame: np.ndarray) -> None:
    draw_skeleton(keypoints_xys, frame)
    draw_bounding_box(keypoints_xys, frame)


def draw_skeleton(keypoints: np.ndarray, frame: np.ndarray) -> None:
    height, width, _ = frame.shape
    keypoint_coordinates = keypoints * np.array([width, height, 1])

    for connection in KEYPOINT_CONNECTIONS:
        start, end = connection
        x1, y1, s1 = keypoint_coordinates[start]
        x2, y2, s2 = keypoint_coordinates[end]
        avg_score = np.average([s1, s2])

        color = get_color_by_score(avg_score)
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)

    for x, y, score in keypoint_coordinates:
        color = get_color_by_score(score)
        cv2.circle(frame, (int(x), int(y)), 9, color, -1)


def draw_bounding_box(keypoints_xys: np.ndarray, frame: np.ndarray) -> None:
    frame_height, frame_width, _ = frame.shape
    scaling_factor = np.array([frame_width, frame_height, 1])

    keypoint_coordinates = np.round(keypoints_xys * scaling_factor).astype(int)
    min_x, min_y, _ = np.min(keypoint_coordinates, axis=0)
    max_x, max_y, _ = np.max(keypoint_coordinates, axis=0)

    avg_score = float(np.mean(keypoints_xys[:, 2]))
    color = get_color_by_score(avg_score)
    thickness = 4
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, thickness)


def crop_center_square(frame: np.ndarray) -> any:
    frame_height, frame_width, _channels = frame.shape
    min_dimension = min(frame_height, frame_width)
    top = (frame_height - min_dimension) // 2
    left = (frame_width - min_dimension) // 2
    cropped_frame = frame[top:top+min_dimension, left:left+min_dimension]
    return cropped_frame


def process_capture_frame(frame: np.ndarray) -> None:
    cropped_frame = crop_center_square(frame)
    frame_height, frame_width, _ = cropped_frame.shape
    prediction, outputs = movenet.predict(cropped_frame)
    keypoints = prediction.squeeze()
    draw_markers(keypoints, cropped_frame)
    cv2.imshow("Capture", frame)

   # Reshape the prediction object to match the model input shape, e.g., [1, 51]
    reshaped_prediction = np.reshape(prediction, (1, -1))

    # Run the TFLite model inference
    output_data = run_tflite_model(
        pose_classifier_interpreter, reshaped_prediction)

    # Post-process the model output, if needed
    process_pose_classifier_output(output_data)


class MoveNet(object):
    __module = None
    __input_size: int = 0
    __model = None

    def __init__(self, model_name: str = "movenet_thunder"):
        self.__init_model(model_name)

    def __init_model(self, model_name: str):
        """using models from tensorhub https://tfhub.dev/s?module-type=image-pose-detection"""
        if "movenet_lightning" in model_name:
            self.__module = hub.load(
                "https://tfhub.dev/google/movenet/singlepose/lightning/4")
            self.__input_size = 192
        elif "movenet_thunder" in model_name:
            self.__module = hub.load(
                "https://tfhub.dev/google/movenet/singlepose/thunder/4")
            self.__input_size = 256
        else:
            raise ValueError("Unsupported model name: %s" % model_name)

        self.__model = self.__module.signatures["serving_default"]
        print(f"movenet initialized with model {model_name}")

    def __process_image(self, input_image: tf.Tensor):
        expanded_image = tf.expand_dims(input_image, axis=0)
        resized_image = tf.image.resize_with_pad(
            expanded_image, self.__input_size, self.__input_size)
        # SavedModel format expects tensor type of int32.
        processed_image = tf.cast(resized_image, dtype=tf.int32)
        return processed_image

    def predict(self, input_image: tf.Tensor) -> np.ndarray:

        model = self.__module.signatures['serving_default']

        # Process Image
        processed_image = self.__process_image(input_image)
        # Run model inference.
        outputs = model(processed_image)
        # Output is a [1, 1, 17, 3] tensor.
        prediction: np.ndarray = outputs['output_0'].numpy()
        # prediction's x and y columns are flipped
        prediction = prediction.squeeze()[:, [1, 0, 2]]
        return prediction, outputs


def capture_video(process_frame):
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        raise RuntimeError("Could not access webcam")

    while True:
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
        ret, frame = capture.read()
        if not ret:
            raise RuntimeError("Failed to read from video capture")
        process_frame(frame)

    capture.release()
    cv2.destroyAllWindows()


# Load TFLite model
pose_classifier_tflite_path = "model.tflite"
pose_classifier_interpreter = load_tflite_model(pose_classifier_tflite_path)

movenet = MoveNet(model_name="movenet_lightning")
capture_video(process_capture_frame)
