import bpy
import os
import math


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def add_track_to_constraint(camera, target):
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'


def create_collection(collection_name):
    if collection_name not in bpy.data.collections:
        new_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_collection)
    else:
        new_collection = bpy.data.collections[collection_name]
    return new_collection


def create_and_setup_cameras(cameras, empty_object, collection):
    for name, location, rotation, focal_length in cameras:
        # Check if the camera with the specified name already exists
        camera = bpy.data.objects.get(name)

        if camera is None:
           # If the camera doesn't exist, create a new one
            camera_data = bpy.data.cameras.new(name)
            camera_data.lens = focal_length
            camera = bpy.data.objects.new(name, camera_data)
            camera.location = location
            camera.rotation_euler = rotation

            # Add the camera to the specified collection
            if camera.name not in collection.objects:
                collection.objects.link(camera)
        else:
            # If the camera exists, update its location, rotation, and focal length
            camera.location = location
            camera.rotation_euler = rotation
            camera.data.lens = focal_length

        # Add the 'Track To' constraint to the camera, if it's not already present
        if not any(c.type == 'TRACK_TO' for c in camera.constraints):
            add_track_to_constraint(camera, empty_object)


def create_empty_object(location, collection):
    empty_name = "Empty"
    empty_object = bpy.data.objects.get(empty_name)

    if empty_object is None:
        # Create a new empty object
        empty_object = bpy.data.objects.new(empty_name, None)
        empty_object.location = location

        # Add the empty object to the specified collection
        if empty_object.name not in collection.objects:
            collection.objects.link(empty_object)

    return empty_object


def generate_cameras(num_cameras, radius, heights, focal_length):
    all_cameras = []
    camera_count = 0

    for height in heights:
        angle_step = 2 * math.pi / num_cameras
        for i in range(num_cameras):
            angle = i * angle_step
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = height

            location = (x, y, z)
            rotation = (0, 0, angle)
            camera_name = f"Camera{camera_count + 1}"

            all_cameras.append((camera_name, location, rotation, focal_length))
            camera_count += 1

    return all_cameras


def ensure_output_directory_exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)


def set_render_settings(file_format, resolution_x, resolution_y):
    bpy.context.scene.render.image_settings.file_format = file_format
    bpy.context.scene.render.resolution_x = resolution_x
    bpy.context.scene.render.resolution_y = resolution_y


def apply_bone_scales(armature_name, selected_body_type_scales):
    # Access the armature object
    armature = bpy.data.objects[armature_name]

    # Ensure the armature is selected
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    # Iterate over body types, applying the scaling factors
    # Make sure the armature is in pose mode
    bpy.ops.object.mode_set(mode='POSE')

    # Apply the scale changes to the specified bones
    for bone_name, scale_factors in selected_body_type_scales.items():
        bone = armature.pose.bones[bone_name]
        bone.scale.x = scale_factors[0]
        bone.scale.y = scale_factors[1]
        bone.scale.z = scale_factors[2]

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')


def render_and_save_images(output_directory, camera_names, body_types, armature_name, focal_lengths):
    total_images = len(camera_names) * len(body_types) * len(focal_lengths)
    current_image_count = 0

    # Iterate over body types, applying the scaling factors, rendering, and saving the images
    for selected_body_type in body_types.keys():
        # Pass the dictionary instead of the tuple
        apply_bone_scales(armature_name, body_types[selected_body_type])

        # Render and save images from each camera's viewpoint
        for camera_name in camera_names:
            bpy.context.scene.camera = bpy.data.objects[camera_name]

            for focal_length in focal_lengths:
                bpy.context.scene.camera.data.lens = focal_length
                bpy.context.scene.render.filepath = os.path.join(
                    output_directory, f"{selected_body_type}_{camera_name}_F{focal_length}.jpg")
                bpy.ops.render.render(write_still=True)

                # Prefix for profress bar
                prefix = f"Image Progress {current_image_count}/{total_images}:"
                # Update the progress bar
                current_image_count += 1
                printProgressBar(current_image_count, total_images,
                                 prefix=prefix, suffix='Complete', length=50)


def main():

    num_cameras = 1
    radius = 3.5
    # num_camers for each height, so 3 heighs is 150 cameras
    heights = [0.5, 1.0, 1.5]
    focal_lengths = [25, 35, 50]
    cameras = generate_cameras(num_cameras, radius, heights, focal_lengths[1])

    # List the names of the cameras you have manually placed in the scene
    camera_names = [camera[0] for camera in cameras]

    # Set the output directory for the rendered images
    output_directory = r"C:\Users\Admin\Documents\Python\TensorFlow\Keras\Pose Classification\images"

    # Define preset scale factors for different body types
    # (X, Y, Z)
    body_types = {
        "ectomorph": {
            "neck": (0.9, 0.9, 1.0),
            "spine": (0.8, 0.8, 0.7),
            "spine.001": (0.8, 0.8, 0.8),
            "shoulder.L": (0.9, 0.9, 0.9),
            "shoulder.R": (0.9, 0.9, 0.9),
            "upper_arm.L": (0.9, 0.9, 1.1),
            "upper_arm.R": (0.9, 0.9, 1.1),
            "forearm.L": (0.9, 1.0, 1.2),
            "forearm.R": (0.9, 1.0, 1.2),
            "thigh.L": (0.8, 0.85, 1.1),
            "thigh.R": (0.8, 0.85, 1.1),
            "shin.L": (0.9, 1.15, 1.3),
            "shin.R": (0.9, 1.15, 1.3),
        },
        "mesomorph": {
            "neck": (1.0, 1.0, 1.0),
            "spine": (1.0, 1.0, 1.0),
            "spine.001": (1.0, 1.0, 1.0),
            "shoulder.L": (1.0, 1.0, 1.0),
            "shoulder.R": (1.0, 1.0, 1.0),
            "upper_arm.L": (1.0, 1.0, 1.0),
            "upper_arm.R": (1.0, 1.0, 1.0),
            "forearm.L": (1.0, 1.0, 1.0),
            "forearm.R": (1.0, 1.0, 1.0),
            "thigh.L": (1.0, 1.0, 1.0),
            "thigh.R": (1.0, 1.0, 1.0),
            "shin.L": (1.0, 1.0, 1.0),
            "shin.R": (1.0, 1.0, 1.0),
        },
        "endomorph": {
            "neck": (1.0, 1.0, 1.0),
            "spine": (1.0, 1.0, 1.0),
            "spine.001": (1.0, 1.2, 1.0),
            "spine.002": (1.0, 1.05, 1.0),
            "shoulder.L": (1.1, 1.1, 1.0),
            "shoulder.R": (1.1, 1.1, 1.0),
            "upper_arm.L": (1.0, 0.95, 1.0),
            "upper_arm.R": (1.0, 0.95, 1.0),
            "forearm.L": (1.0, 1.0, 1.0),
            "forearm.R": (1.0, 1.0, 1.0),
            "thigh.L": (1.0, 1.1, 1.0),
            "thigh.R": (1.0, 1.1, 1.0),
            "shin.L": (1.0, 0.9, 1.0),
            "shin.R": (1.0, 0.9, 1.0),
        },
    }

   # Create a new collection for cameras and the empty object
    camera_collection = create_collection("Camera_Empty_Object")
    empty_object = create_empty_object((0, 0, 0.6), camera_collection)
    create_and_setup_cameras(cameras, empty_object, camera_collection)
    ensure_output_directory_exists(output_directory)
    set_render_settings('JPEG', 1920, 1080)
    render_and_save_images(
        output_directory, camera_names, body_types, "HG_Jamie", focal_lengths)


if __name__ == "__main__":
    main()
