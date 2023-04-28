import bpy

# Set the name of the armature object
armature_name = "HG_Allen"

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

# Choose the desired body type
selected_body_type = "endomorph"

# Access the armature object
armature = bpy.data.objects[armature_name]

# Make sure the armature is in pose mode
bpy.ops.object.mode_set(mode='POSE')

# Apply the scale changes to the specified bones
for bone_name, scale_factors in body_types[selected_body_type].items():
    bone = armature.pose.bones[bone_name]
    bone.scale.x = scale_factors[0]
    bone.scale.y = scale_factors[1]
    bone.scale.z = scale_factors[2]
    print(scale_factors[2])

# Return to object mode
bpy.ops.object.mode_set(mode='OBJECT')
