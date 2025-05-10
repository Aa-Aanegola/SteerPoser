import sys
sys.path.append('/workspace/SteerKep/activation-steering')

from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import MalleableModel, SteeringVector
import torch
import re

import re

import re

def parse_output(raw_text):
    """
    Extracts valid Python code from raw model output.
    Keeps f-strings and complex expressions.
    Removes special tokens, system/user metadata, and dangling fragments.
    """
    # Remove special tokens like <|eot_id|>, <|start_header_id|>, etc.
    cleaned = re.sub(r'<\|.*?\|>', '', raw_text)

    # Remove dangling incomplete tokens (e.g. lines with just "com" or nonsense)
    lines = [line.rstrip() for line in cleaned.split('\n')]
    valid_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Filter out likely incomplete or broken lines
        if len(line) < 5 and not re.match(r'\w+\(.*\)', line):
            continue
        valid_lines.append(line)

    return '\n'.join(valid_lines)


steering_map = {
    "sammy" : ["junk-healthy-24b.svec", -2.0, [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]],
    "aakash" : ["junk-healthy-24b.svec", 2.0, [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]],
    "danelle" : ["junk-healthy-24b.svec", 1.0, [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]],
    "default" : None, # could set a default safety steering vector or something. 
}


class UserSteeredModel:
    def __init__(self, cfg, user, verbose=True):
        self.cfg = cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose:
            print(f"Loading {cfg.model_name} into HF cache dir {cfg.cache_dir} on device {device}")
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir, local_files_only=True, device_map='auto', torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir, local_files_only=True)
        self.malleable_model = MalleableModel(self.model, self.tokenizer)
        self.settings = {
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": False,
            "max_new_tokens": cfg.max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        self.model.eval()
        self.use_steering = False

    def set_user(self, user):
        if user in steering_map.keys():
            vec, strength, layer_ids = steering_map[user]
            self.steering_vector_name = vec
            self.use_steering = True
            self.update_steer(vec, strength, behavior_layer_ids)
        elif(steering_map['default'] is not None):
            vec, strength, layer_ids = steering_map['default']
            self.steering_vector_name = vec
            self.use_steering = True
            self.update_steer(vec, strength, behavior_layer_ids)

    def update_steer(self, steer_vector, strength, behavior_layer_ids):
        self.malleable_model = MalleableModel(self.model, self.tokenizer)
        svpath = os.path.join(cfg.steering_vector_dir, steer_vector)
        self.steering_vector = SteeringVector.load(svpath)
        self.malleable_model.steer(
            self.steering_vector,
            behavior_layer_ids= behavior_layer_ids,
            behavior_vector_strength=strength,
        )
        
    def generate(self, prompts, debug=False):
        # Accept a single string or a list of strings
        if isinstance(prompts, str):
            prompts = [prompts]

        if self.use_steering:
            if(debug):
                print(f"Steering with {self.steering_vector_name}\n")
            # Steered generation using MalleableModel
            responses = self.malleable_model.respond_batch_sequential(prompts=prompts,settings=self.settings) 
        else:
             # Unsteered generation using regular Hugging Face model
             inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
             outputs = self.model.generate(**inputs, **self.settings)
             responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # # If user passed a single prompt, return a single string
        # ret = []
        # for response in responses:
        #     ret.append(response.split('#')[0])

        # print("from the steered model -", ret)
        if(debug):
            print("raw output: ", responses[0], "\n\n")
        responses = [parse_output(txt) for txt in responses]
        responses = responses[0] if len(responses) == 1 else responses
        return responses


if __name__ == '__main__':
    from arguments import get_config
    cfg = get_config(config_path='./configs/steering.yaml')
    model = SteeredModel(cfg)
    print(model.generate("write a haiku about frogs"))

    
    
