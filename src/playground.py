import os
import subprocess
import time

import sys
sys.path.append('/workspace/SteerKep')
sys.path.append('/workspace/SteerKep/RLBench')

# Start Xvfb manually (xvfb-run does this under the hood)
subprocess.Popen([
    'Xvfb', ':99', '-screen', '0', '1280x1024x24', '+extension', 'GLX', '+render', '-noreset'
])

time.sleep(2)  # Give Xvfb time to start

# Set environment variables to match your working terminal setup
os.environ['DISPLAY'] = ':99'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
# os.environ['LIBGL_DRIVERS_PATH'] = '/usr/lib/x86_64-linux-gnu/dri'
# os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libffi.so.7'

import openai
os.environ["OPENAI_API_KEY"] = "hehe boi" 
from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from envs.rlbench_env import VoxPoserRLBench
from utils import set_lmp_objects
import numpy as np
from rlbench import tasks
from steered_model import SteeredModel

  # set your API key here
  
config = get_config('rlbench')
# uncomment this if you'd like to change the language model (e.g., for faster speed or lower cost)
# for lmp_name, cfg in config['lmp_config']['lmps'].items():
#     cfg['model'] = 'gpt-3.5-turbo'

# initialize env and voxposer ui
visualizer = ValueMapVisualizer(config['visualizer'])
env = VoxPoserRLBench(visualizer=visualizer)

steering_cfg = get_config(config_path='./configs/steering.yaml')
model = SteeredModel(steering_cfg)

lmps, lmp_env = setup_LMP(env, config, debug=False, model=model)
voxposer_ui = lmps['plan_ui']

# below are the tasks that have object names added to the "task_object_names.json" file
# uncomment one to use
# env.load_task(tasks.PutRubbishInBin)
# env.load_task(tasks.LampOff)
# env.load_task(tasks.OpenWineBottle)
# env.load_task(tasks.PushButton)
# env.load_task(tasks.TakeOffWeighingScales)
# env.load_task(tasks.MeatOffGrill)
# env.load_task(tasks.SlideBlockToTarget)
# env.load_task(tasks.TakeLidOffSaucepan)
# env.load_task(tasks.TakeUmbrellaOutOfUmbrellaStand)
env.load_task(tasks.SetTheTable)

print("MOMENT OF TRUTH")
#env.task.reset()
print("INVICTUSSSSS")
descriptions, obs = env.reset()
#print(descriptions, obs)
set_lmp_objects(lmps, env.get_object_names())
instruction = np.random.choice(descriptions)
voxposer_ui(instruction)
