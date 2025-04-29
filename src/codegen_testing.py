import os
import subprocess
import time
import sys
sys.path.append('/workspace/activation-steering')
# Start Xvfb manually (xvfb-run does this under the hood)
subprocess.Popen([
    'Xvfb', ':99', '-screen', '0', '1280x1024x24', '+extension', 'GLX', '+render', '-noreset'
])
time.sleep(2)  # Give Xvfb time to start

# Set environment variables to match your working terminal setup
os.environ['DISPLAY'] = ':99'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
import openai
from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from envs.rlbench_env import VoxPoserRLBench
from utils import set_lmp_objects
import numpy as np
from rlbench import tasks
from steered_model import SteeredModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import MalleableModel, SteeringVector

config = get_config('rlbench')

# for lmp_name, cfg in config['lmp_config']['lmps'].items():
#     cfg['model'] = 'gpt-3.5-turbo', 'Hermes'

visualizer = ValueMapVisualizer(config['visualizer'])
env = VoxPoserRLBench(visualizer=visualizer)
print("done env and visualizer")

steer_cfg = get_config(config_path='./configs/steering.yaml')
model = SteeredModel(self.steer_cfg)
print("done loading model")

lmps, lmp_env = setup_LMP(env, config, debug=False)
print(f"Set up lmps: {lmps.keys()}")

