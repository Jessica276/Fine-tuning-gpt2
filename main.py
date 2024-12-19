import torch
from torch.utils.data import DataLoader
from preprocess import load_and_preprocess_data
from data_handler import DataHandler
from model import load_model_and_tokenizer
from train import train_model
from generate import generate_text

def main():
    text_train, text_val = load_and_preprocess_data()
    
    max_seq_len = 50

    tokenizer,model = load_model_and_tokenizer(model_name="openai-community/gpt2")
    train_dataset = DataHandler(text_train["text"], max_seq_len, tokenizer)
    eval_dataset = DataHandler(text_val["text"], max_seq_len, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(device,tokenizer,model,train_dataset,eval_dataset,max_seq_len)

    # Generate text
    input_text = "Misy fomba mahomby"
    generated_text = generate_text(model, tokenizer, input_text, device)
    print(generated_text)


if __name__ == "__main__":
    main()
