import bpy


def add_track_to_constraint(camera, target):
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'


# Create an empty object at the center of the scene
bpy.ops.object.add(type='EMPTY', location=(0, 0, 0.6))
empty_object = bpy.context.active_object

# Camera name: The name you want to assign to the camera object in Blender
# Location: A tuple (x, y, z) representing the 3D location of the camera in the scene
# Rotation: A tuple (x, y, z) representing the rotation of the camera (in radians)
# Focal length: The focal length of the camera lens in millimeters (optional)
cameras = [
    ("Camera1", (-3.18261, -3.45106, 1.14772), (0, 0, 0), 35),
    ("Camera2", (3.18261, -3.45106, 1.14772), (0, 0, 0), 35),
    ("Camera3", (-3.18261, -3.45106 / 2, 1.14772), (0, 0, 0), 35),
    ("Camera4", (3.18261, -3.45106 / 2, 1.14772), (0, 0, 0), 35),
    ("Camera5", (0, -3.45106, 0.5), (0, 0, 0), 35),
    ("Camera6", (0, -3.45106, 1.0), (0, 0, 0), 35),
]

# Create and set up cameras
for name, location, rotation, focal_length in cameras:
    bpy.ops.object.camera_add(location=location, rotation=rotation)
    camera = bpy.context.active_object
    camera.name = name
    camera.data.lens = focal_length

    # Add the 'Track To' constraint to the camera
    add_track_to_constraint(camera, empty_object)
