# from model import create_and_train_model
import tensorflow as tf
from pose_estimation.pose_estimation_preprocessing import example_test
from data_preparation.generate_datasets import generate_set
from data_preparation.data_splitting import split_init
from pose_estimation.pose_data_types import BodyPart
from models.train_model import create_and_train_model
from models.convert_model import convert_model
import sys
import os

base_key_points = [BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP,
                   BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER]

exercise_key_points = {
    "squat": {
        "image_set_dir": 'C:\\Users\\Admin\\Documents\\Python\\TensorFlow\\Keras\\Pose Classification\\python_keras_model\\data\\squat\\squat_poses',
        "keypoints": base_key_points + [BodyPart.LEFT_KNEE, BodyPart.RIGHT_KNEE, BodyPart.LEFT_ANKLE, BodyPart.RIGHT_ANKLE]
        },
}

exercise = "squat"
key_points = exercise_key_points[exercise]["keypoints"]
key_point_indices = [kp.value for kp in key_points]
dataset_in = exercise_key_points[exercise]["image_set_dir"]
# Create a folder for each exercise's model
model_output_dir = f'./models/{exercise}_model'
os.makedirs(model_output_dir, exist_ok=True)

root = split_init(dataset_in)
print(root)
csvs_out_test_path, csvs_out_train_path = generate_set(root, train_folder='train', test_folder='test', key_point_indices=key_point_indices)

# Check if tfjs model already exists
tfjs_model_path = os.path.join(model_output_dir, 'model.json')
if not os.path.isfile(tfjs_model_path):    
    model, history = create_and_train_model(csvs_out_test_path=csvs_out_test_path, csvs_out_train_path=csvs_out_train_path)
    convert_model(model, model_output_dir=model_output_dir)
else:
    print(f"Model '{tfjs_model_path}' already exists, skipping training and conversion.")