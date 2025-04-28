from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import MalleableModel, SteeringVector

class SteeredModel:
    def __init__(self, cfg):
        self.cfg = cfg
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
        steered_responses = malleable_model.respond_batch_sequential(
            prompts=[prompt],
            use_chat_template=False,
            settings = settings
        )
        return steered_responses[0]