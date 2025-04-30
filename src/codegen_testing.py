import os
import subprocess
import time
import sys
# sys.path.append('~/activation-steering')

if not os.path.exists("/tmp/.X99-lock"):
    # Start Xvfb manually (xvfb-run does this under the hood)
    xvfb_proc = subprocess.Popen([
        'Xvfb', ':99', '-screen', '0', '1280x1024x24', '+extension', 'GLX', '+render', '-noreset'
    ])
    print(f"Started xvfb at PID {xvfb_proc.pid}")
    
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
env.load_task(tasks.SetTheTable)
print("done env and visualizer")

steer_cfg = get_config(config_path='./configs/steering.yaml')
model = SteeredModel(steer_cfg)
print("done loading model")

lmps, lmp_env = setup_LMP(env, config, debug=True, model=model)
print(f"Set up lmps: {lmps.keys()}")

descriptions, obs = env.reset()
instruction = np.random.choice(descriptions)
print(f"Chose from {descriptions}, selected {instruction}")
print("\n___________________________\n", obs)
set_lmp_objects(lmps, env.get_object_names())  # set the object names to be used by voxposer
affordance_mapper = lmps['get_affordance_map']

#lmps['plan_ui'](instruction)

prompt, user_query = self.build_prompt(query)
code_str, prompt = affordance_mapper._local_call(
    prompt=prompt,
    stop=affordance_mapper._stop_tokens,
    temperature=affordance_mapper._cfg['temperature'],
    max_tokens=affordance_mapper._cfg['max_tokens']
)
print(f'*** Local call took {time.time() - start_time:.2f}s ***')

print(f"LLM with {prompt=} generated {code_str=}")


