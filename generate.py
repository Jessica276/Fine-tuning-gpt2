def generate_text(model, tokenizer, input_text, device, max_length=100):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)
