import sys
sys.path.append('/workspace/SteerKep/activation-steering')
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import SteeringDataset, MalleableModel, SteeringVector
import torch
import re
import os

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
    "sammy" : dict(sv_name="junk-healthy-24b.svec", strength=-2.0, hidden_layer_ids=[29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]),
    "aakash" : dict(sv_name="junk-healthy-24b.svec", strength=2.0, hidden_layer_ids=[29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]),
    "danelle" : dict(sv_name="junk-healthy-24b.svec", strength=1.0, hidden_layer_ids=[29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]),
    "default" : None, # could set a default safety steering vector or something. 
}


class UserSteeredModel:
    def __init__(self, cfg, user, model=None, tokenizer=None, use_chat_template=False, verbose=True):
        self.cfg = cfg
        assert user in steering_map, f"{user} not recognized user, must be one of steering_map.keys()"
        self.user = user
        self.user_metadata = steering_map[user]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose:
            print(f"Loading {cfg.model_name} into HF cache dir {cfg.cache_dir} on device {device}")
        if(model is None):
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir, local_files_only=True, device_map='auto', torch_dtype=torch.float16)
        else:
            self.model = model
        if(tokenizer is None):
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir, local_files_only=True)
        else:
            self.tokenizer = tokenizer
        self.malleable_model = MalleableModel(self.model, self.tokenizer)
        self.settings = {
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": False,
            "max_new_tokens": cfg.max_new_tokens,
            "repetition_penalty" : 1.2, 
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        self.use_chat_template = use_chat_template
        self.model.eval()
        self.use_steering = False
        self.steering_vector = None

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

    def set_steer_vector(self, steer_vector_name):
        print(f"setting {self.user} user model to steer vec {steer_vector_name}")
        svpath = os.path.join(self.cfg.steering_vector_dir, steer_vector_name)
        self.steering_vector = SteeringVector.load(svpath)

    def update_sv_strength_by_ratings(self, ratings_fname, sv_name=None, norm_H=False, eps=1e-8):
        if(sv_name is not None):
            self.set_steer_vector(sv_name)
        else:
            assert self.steering_vector is not None, "With no sv provided, must already be saved to model"
        
        with open(os.path.join(self.cfg.steering_datasets_dir, ratings_fname), 'r') as f:
            dset = json.load(f)
        examples = []
        suffixes = []
        for item in dset:
            examples.append((item["input"], item["input"]))
            suffixes.append((item["compliant_continuation"], item["non_compliant_continuation"]))
        self.ratings_dset = SteeringDataset(self.tokenizer, examples, suffixes, use_chat_template=self.use_chat_template) 
        print("Successfully reloaded ratings dataset")
        hidden_layer_ids = self.user_metadata['hidden_layer_ids']
        strengths, layer_hiddens, train_strs = self.steering_vector.fit_sv_strength(self.malleable_model, self.tokenizer, self.ratings_dset, hidden_layer_ids, norm_H=norm_H)
        layer_ratings = []
        for id, s in strengths.items():
            pos, neg = s[::2].mean(), s[1::2].mean()
            rating = (pos - neg) / (abs(pos) + abs(neg) + eps)
            print(f"Raw rating for layer {id} is {rating}")
            layer_ratings.append(rating)
        return(layer_ratings, strengths, layer_hiddens, train_strs)
        
    def apply_steering(self, steer_vector_name, strength, behavior_layer_ids):
        self.malleable_model = MalleableModel(self.model, self.tokenizer)
        self.malleable_model.steer(
            self.steering_vector,
            behavior_layer_ids= behavior_layer_ids,
            behavior_vector_strength=strength,
        )

    def generate(self, prompts, temp=0.1, debug=False):
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
            if(self.use_chat_template):
                messages = [dict(role="system", content=p) for p in prompts]
                inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(self.model.device)
            else:
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            outputs = self.model.generate(**inputs, **self.settings)
            responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_space=True)

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

    
    
