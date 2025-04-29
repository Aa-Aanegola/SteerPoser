import os
import subprocess
import time

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
os.environ["OPENAI_API_KEY"] = "Key Here"
from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from envs.rlbench_env import VoxPoserRLBench
from utils import set_lmp_objects
import numpy as np
from rlbench import tasks

  # set your API key here
  
config = get_config('rlbench')
# uncomment this if you'd like to change the language model (e.g., for faster speed or lower cost)
# for lmp_name, cfg in config['lmp_config']['lmps'].items():
#     cfg['model'] = 'gpt-3.5-turbo'

# initialize env and voxposer ui
visualizer = ValueMapVisualizer(config['visualizer'])
env = VoxPoserRLBench(visualizer=visualizer)
# lmps, lmp_env = setup_LMP(env, config, debug=False)
# voxposer_ui = lmps['plan_ui']

# below are the tasks that have object names added to the "task_object_names.json" file
# uncomment one to use
# env.load_task(tasks.PutRubbishInBin)
# env.load_task(tasks.OpenMicrowave)
# env.load_task(tasks.OpenWineBottle)
# env.load_task(tasks.PushButton)
# env.load_task(tasks.TakeOffWeighingScales)
# env.load_task(tasks.MeatOffGrill)
# env.load_task(tasks.SlideBlockToTarget)
# env.load_task(tasks.TakeLidOffSaucepan)
# env.load_task(tasks.TakeUmbrellaOutOfUmbrellaStand)

#task 1
env.load_task(tasks.OpenMicrowave)
env.load_task(tasks.PickAndLift)
env.load_task(tasks.CloseMicrowave)
env.load_task(tasks.PushButton)

#task 2
env.load_task(tasks.OpenOven)
env.load_task(tasks.PickAndLift)
env.load_task(tasks.PutTrayInOven)

#task 3
env.load_task(tasks.PutKnifeInKnifeBlock)
env.load_task(tasks.PutKnifeOnChoppingBoard)


descriptions, obs = env.reset()
# set_lmp_objects(lmps, env.get_object_names())  # set the object names to be used by voxposer