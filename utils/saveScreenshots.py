import bpy
import os

# Set the output directory for the rendered images
output_directory = r"C:\Users\Admin\Documents\Python\TensorFlow\Keras\Pose Classification\images"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# List the names of the cameras you have manually placed in the scene
camera_names = ["Camera1", "Camera2",
                "Camera3", "Camera4", "Camera5", "Camera6"]

# Set the render settings
bpy.context.scene.render.image_settings.file_format = 'JPEG'
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080

# Render and save images from each camera's viewpoint
for camera_name in camera_names:
    bpy.context.scene.camera = bpy.data.objects[camera_name]
    bpy.context.scene.render.filepath = os.path.join(
        output_directory, f"{camera_name}.jpg")
    bpy.ops.render.render(write_still=True)
