import os
import matplotlib.pyplot as plt
import numpy as np
from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.backend import sim
from pyrep.objects.shape import Shape
from pyrep.objects.light import Light


import numpy as np

def look_at(camera_position, target_position):
    direction = np.array(target_position) - np.array(camera_position)
    direction /= np.linalg.norm(direction)
    # Default camera orientation (pointing down z)
    up = np.array([0, 0, 1])
    right = np.cross(up, direction)
    up = np.cross(direction, right)

    import math
    # Convert direction vector to Euler angles
    yaw = np.arctan2(direction[1], direction[0])
    pitch = np.arcsin(-direction[2])
    roll = 0.0
    return [roll, pitch, yaw]

SCENE_PATH = os.path.join(os.getcwd(), 'my_scene.ttt')

os.environ['DISPLAY'] = ':99'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

pr = PyRep()
pr.launch(SCENE_PATH, headless=True)
pr.start()

light = Light.create()
light.set_position([center_pos[0] + 1.0, center_pos[1] + 1.0, center_pos[2] + 2.0])
light.set_orientation([-0.5, 0.5, 0.0])
light.set_diffuse([1.0, 1.0, 1.0])
light.set_specular([0.5, 0.5, 0.5])

# ✅ Query all shapes in the scene
handles = sim.simGetObjectsInTree(sim.sim_handle_scene, sim.sim_object_shape_type, 0)
all_objects = [Shape(h) for h in handles]
print(f"Found {len(all_objects)} shape objects.")

# Compute scene center
positions = np.array([obj.get_position() for obj in all_objects])
center_pos = np.mean(positions, axis=0)
print(f"Scene center: {center_pos}")

# Place a top-down camera
camera = VisionSensor.create([512, 512])
camera_pos = [center_pos[0], center_pos[1], center_pos[2] + 1.5]
camera.set_position(camera_pos)
camera.set_orientation(look_at(camera_pos, center_pos))

# Render image
image = camera.capture_rgb()
plt.imsave('rendered_scene.png', image)
print("Saved image as rendered_scene.png")

pr.stop()
pr.shutdown()
