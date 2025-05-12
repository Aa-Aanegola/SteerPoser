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
from utils import set_lmp_objects, get_steer_cfg_path
import numpy as np
from rlbench import tasks
from steered_model import SteeredModel
# from user_steered_model import UserSteeredModel

  
config = get_config('rlbench')
# initialize env and voxposer ui
visualizer = ValueMapVisualizer(config['visualizer'])
env = VoxPoserRLBench(visualizer=visualizer)
user = str(input("Hi! I am SteerBot, and I am here to help you. Who am I talking to right now? ")).lower()
steer_cfg_path = get_steer_cfg_path(user)

steering_cfg = get_config(config_path=steer_cfg_path)
model = SteeredModel(steering_cfg, user)
model.set_steer_vector("junk-healthy-24b.svec")

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


feedback = str(input("Please share any feedback on how well I assisted you today. "))




if hidden_layer_ids is None:
    hidden_layer_ids = range(model.config.num_hidden_layers)
tokenizer.pad_token_id = 0
batch_size = 32
accumulate_last_x_tokens = "suffix-only"
suffixes = steering_dataset.suffixes
method = "pca_center"
n_layers = len(get_model_layer_list(model))
# Normalize the layer indexes if they are negative
hidden_layer_ids = [i if i >= 0 else n_layers + i for i in hidden_layer_ids]
train_strs = [s for ex in steering_dataset.formatted_dataset for s in (ex.positive, ex.negative)]
layer_hiddens = batched_get_hiddens(
    model, tokenizer, train_strs, hidden_layer_ids, batch_size, accumulate_last_x_tokens, suffixes
)
strengths = {}
for layer in custom_progress(hidden_layer_ids, description="Reading Hidden Representations ..."):
    H = layer_hiddens[layer]
    steer_vector_direction = self.directions[layer]
    steer_vector_direction_strength = project_onto_direction(H, steer_vector_direction)
    print(f"strength for layer {layer}: {steer_vector_direction_strength}")
    strengths[layer] = steer_vector_direction_strength