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



class SteeredModel:
    def __init__(self, cfg, verbose=True):
        self.cfg = cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose:
            print(f"Loading {cfg.model_name} into HF cache dir {cfg.cache_dir} on device {device}")
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir, local_files_only=True, device_map='auto', torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir, local_files_only=True)
        #main_device = model.hf_device_map.get(next(iter(self.model.hf_device_map)))
        #main_device = torch.device(f"cuda:{main_device}") if isinstance(main_device, int) else torch.device(main_device)
        self.steering_vector = SteeringVector.load(cfg.steering_vector_path)
        self.malleable_model = MalleableModel(self.model, self.tokenizer)
        self.malleable_model.steer(
            self.steering_vector,
            behavior_layer_ids=cfg.behavior_layer_ids,
            behavior_vector_strength=cfg.behavior_vector_strength,
        )
        #self.malleable_model.device = main_device
        self.settings = {
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": False,
            "max_new_tokens": cfg.max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        self.model.eval()

    def generate(self, prompts, steer=True):
        # Accept a single string or a list of strings
        if isinstance(prompts, str):
            prompts = [prompts]

        if steer:
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
        print("raw output: ", responses[0])
        print("\nafter parsing", parse_output(responses[0]))
        return parse_output(responses[0]) if len(responses) == 1 else responses

if __name__ == '__main__':
    from arguments import get_config
    cfg = get_config(config_path='./configs/steering.yaml')
    model = SteeredModel(cfg)
    print(model.generate("write a haiku about frogs"))

    
    
