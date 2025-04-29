from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import MalleableModel, SteeringVector

class SteeredModel:
    def __init__(self, cfg, verbose=True):
        self.cfg = cfg
        if(verbose):
            print(f"Loading {cfg.model_name} into HF cache dir {cfg.cache_dir}")
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.steering_vector = SteeringVector.load(cfg.steering_vector_path)
        self.malleable_model = MalleableModel(self.model, self.tokenizer)
        self.malleable_model.steer(
            self.steering_vector,
            behavior_layer_ids=cfg.behavior_layer_ids,
            behavior_vector_strength=cfg.behavior_vector_strength,
        )
        self.settings = {
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": False,
            "max_new_tokens": cfg.max_new_tokens,
            "repetition_penalty": 1.1,
}
    
    # Takes a list of raw text prompts and generates text
    def generate(self, prompt):
        steered_responses = self.malleable_model.respond_batch_sequential(
            prompts=[prompt],
            use_chat_template=False,
            settings = self.settings
        )
        return steered_responses[0]


if __name__ == '__main__':
    from arguments import get_config
    steer_cfg = get_config(config_path='./configs/steering.yaml')
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir, device_map="auto")
    print(f"Loaded {model=}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir)
    print(f"Loaded {tokenizer=}")

    
    