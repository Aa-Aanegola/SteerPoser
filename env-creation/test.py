import os
import subprocess
import time

# 🧠 START Xvfb BEFORE importing PyRep or anything OpenGL-related
subprocess.Popen([
    'Xvfb', ':99', '-screen', '0', '1280x1024x24', '+extension', 'GLX', '+render', '-noreset'
])
time.sleep(2)
os.environ['DISPLAY'] = ':99'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

# Ensure COPPELIASIM_ROOT is set
coppelia_root = os.environ.get('COPPELIASIM_ROOT')
assert coppelia_root is not None, "Set COPPELIASIM_ROOT"

pr = PyRep()
pr.launch('', headless=True)
pr.start()

# Create a table (cuboid)
table = Shape.create(type=PrimitiveShape.CUBOID,
                     size=[1.0, 1.0, 0.1],
                     color=[0.8, 0.8, 0.8],
                     static=True)
table.set_position([0, 0, 0.05])

# Create a cube
cube = Shape.create(type=PrimitiveShape.CUBOID,
                    size=[0.05, 0.05, 0.05],
                    color=[1.0, 0.0, 0.0],
                    static=False)
cube.set_position([0, 0, 0.1])

# Import Franka Panda robot
model_path = os.path.join(coppelia_root, 'models', 'robots', 'non-mobile', 'FrankaEmikaPanda.ttm')
panda = pr.import_model(model_path)
panda.set_position([0, 0, 0])  # move root of model

# Export the scene
scene_path = os.path.join(os.getcwd(), 'my_scene.ttt')
pr.export_scene(scene_path)
print(f"Scene exported to: {scene_path}")

pr.stop()
pr.shutdown()