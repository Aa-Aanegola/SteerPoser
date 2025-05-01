#COPY 

# import os
# import subprocess
import time

# import sys
# sys.path.append('/workspace/SteerKep')
# sys.path.append('/workspace/SteerKep/RLBench')

# # Start Xvfb manually (xvfb-run does this under the hood)
# subprocess.Popen([
#     'Xvfb', ':99', '-screen', '0', '1280x1024x24', '+extension', 'GLX', '+render', '-noreset'
# ])

# time.sleep(2)  # Give Xvfb time to start

# # Set environment variables to match your working terminal setup
# os.environ['DISPLAY'] = ':99'
# os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
# # os.environ['LIBGL_DRIVERS_PATH'] = '/usr/lib/x86_64-linux-gnu/dri'
# # os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libffi.so.7'

import threading
import faulthandler

def dump_stack_after_timeout(seconds=60):
    def dump():
        time.sleep(seconds)
        print(f"Timeout reached ({seconds}s) — dump stack trace yeehaw\n")
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    threading.Thread(target=dump, daemon=True).start()

dump_stack_after_timeout(60)


# import openai
# os.environ["OPENAI_API_KEY"] = "Key Here"
from arguments import get_config
from visualizers import ValueMapVisualizer
from envs.rlbench_env_Copy1 import VoxPoserRLBench1
import numpy as np
from rlbench import tasks

  
config = get_config('rlbench')
# uncomment this if you'd like to change the language model (e.g., for faster speed or lower cost)
# for lmp_name, cfg in config['lmp_config']['lmps'].items():
#     cfg['model'] = 'gpt-3.5-turbo'

# initialize env and voxposer ui
visualizer = ValueMapVisualizer(config['visualizer'])
env = VoxPoserRLBench1(visualizer=None)

env.load_task(tasks.SetTheTable)
env.task._validate = False

print("MOMENT OF TRUTH")
env.task.reset()
print("INVICTUSSSSS")
descriptions, obs = env.reset()
print(descriptions, obs)

# from interfaces import setup_LMP
# from utils import set_lmp_objects
# from steered_model import SteeredModel

# steering_cfg = get_config(config_path='/workspace/SteerKep/SteerPoser/src/configs/steering.yaml')
# model = SteeredModel(steering_cfg)


# lmps, lmp_env = setup_LMP(env, config, debug=False, model=model)
# voxposer_ui = lmps['plan_ui']
# set_lmp_objects(lmps, env.get_object_names())
# instruction = np.random.choice(descriptions)
# voxposer_ui(instruction)
