import torch
from torch import optim


def train_model(device,tokenizer,model,train_dataset,eval_dataset,max_seq_len):
    # Weights and bias
    wandb.login(key="3ffb1629bb5d0897d96a68d99f03ae29812abf2f")
    wandb.init(project="gpt2-malagasy", name="fine-tuning-gpt2")

    # Define hyperparameters
    batch_size = 32
    learning_rate = 1e-4
    epochs = 2


    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "accumulation_steps": 4,
        "optimizer": "Adam",
        "scheduler": "StepLR"
    }


    training_args = TrainingArguments(
        output_dir="/",
        evaluation_strategy="epoch",          
        learning_rate=learning_rate,          
        per_device_train_batch_size=batch_size,       
        per_device_eval_batch_size=batch_size,        
        num_train_epochs=epochs,              
        weight_decay=0.01,                    
        logging_dir="./",                 
        logging_steps=10,                     
        report_to="wandb",                    
        save_strategy="epoch",                
        save_total_limit=2                    
    )

    trainer = Trainer(
        model=model.to(device),                         
        args=training_args,                   
        train_dataset=train_dataset,          
        eval_dataset=eval_dataset,            
        tokenizer=tokenizer
    )

    trainer.train()
    wandb.finish()

    return model