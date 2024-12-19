from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_name="openai-community/gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    return tokenizer, model
