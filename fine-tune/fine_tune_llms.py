import json
import pandas as pd
import torch
import argparse
from datasets import Dataset, load_dataset, DatasetDict
from huggingface_hub import notebook_login
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
import time
import wandb
import transformers
from datetime import datetime

def login_wandb():
    """Log in to Weights & Biases."""
    wandb.login()

def read_data(file_path):
    """Read text data from a file."""
    data = []
    with open(file_path, 'r') as f:
        for l in f.readlines():
            data.append(l.strip())
    return data

def combine_and_zip(text_data, triples_data):
    """Combine and zip text and triples data into tuples."""
    return list(zip(text_data, triples_data))

def create_dataframe(data_list, columns):
    """Create a pandas DataFrame from a list of tuples."""
    return pd.DataFrame(data_list, columns=columns)

def save_to_csv(dataframe, file_name):
    """Save DataFrame to a CSV file."""
    dataframe.to_csv(file_name, index=False)

def load_dataset_from_csv(file_path, split='train'):
    """Load a dataset from a CSV file."""
    return load_dataset("csv", data_files=file_path, split=split)

def load_model_and_tokenizer(model_id, bnb_config):
    """Load the base language model for completion and the tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=512,
        padding_side="left",
        add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def tokenize_prompt(tokenizer, prompt):
    """Tokenize a prompt using the specified tokenizer."""
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point, tokenizer):
    """Generate and tokenize a prompt from a data point."""
    full_prompt = "Transform the text into a semantic graph, which means, extract the triples from the text in format of lists like the following, [[\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"]].\nText: " + data_point['in_txt'] + "\nSemantic graph: " + data_point['triples']
    return tokenize_prompt(tokenizer, full_prompt)

def prepare_model_for_kbit_training(model):
    """Enable gradient checkpointing in the model and prepare for k-bit training."""
    model.gradient_checkpointing_enable()
    return model

def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def configure_lora(model, config):
    """Configure LORA (LOng-Range dependencies in Recurrent Architectures) parameters using LoraConfig."""
    return get_peft_model(model, config)

def check_gpu_count():
    """Check if more than 1 GPU is available and update model properties accordingly."""
    return torch.cuda.device_count() > 1

def configure_trainer(model, train_dataset, val_dataset, tokenizer):
    """Configure the Trainer for training the model."""
    project = "webNLG2020-finetune"
    base_model_name = "llama2-13b"
    run_name = base_model_name + "-" + project
    output_dir = "./" + run_name
    tokenizer.pad_token = tokenizer.eos_token

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=5,
            dataloader_num_workers=2,
            dataloader_prefetch_factor=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=500,
            learning_rate=2.5e-5,
            logging_steps=50,
            bf16=True,
            fp16=False,
            optim="paged_adamw_8bit",
            logging_dir="./logs",
            save_strategy="steps",
            save_steps=50,
            evaluation_strategy="steps",
            eval_steps=50,
            do_eval=True,
            report_to="wandb",
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    return trainer

def train_language_model(base_model_id, train_tgt, train_src, val_tgt, val_src):
    """Train the language model."""
    
    # Log in to Weights & Biases
    login_wandb()
    
    
    # Read training text and triples data
    #text = read_data('webnlg_data/train_val_data/train.target')
    #trpl = read_data('webnlg_data/train_val_data/train.source')
    text = read_data(train_tgt)
    trpl = read_data(train_src)


    # Read validation text and triples data
    #text_val = read_data('webnlg_data/train_val_data/val.target')
    #trpl_val = read_data('webnlg_data/train_val_data/val.source')
    text_val = read_data(val_tgt)
    trpl_val = read_data(val_src)

    # Combine text and triples data into lists
    list_t = combine_and_zip(text, trpl)
    list_v = combine_and_zip(text_val, trpl_val)

    # Create DataFrames for training and validation data
    df = create_dataframe(list_t, columns=['in_txt', 'triples'])
    df_val = create_dataframe(list_v, columns=['in_txt', 'triples'])

    # Save DataFrames to CSV files
    save_to_csv(df, 'train.csv')
    save_to_csv(df_val, 'val.csv')

    # Load training and validation datasets from CSV files
    dataset_train = load_dataset_from_csv("train.csv")
    dataset_val = load_dataset_from_csv("val.csv")

    # Define the base model ID for the Language Model (LLM)
    #base_model_id = "meta-llama/Llama-2-13b-hf"

    # Configure quantization using BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the base language model for completion and the tokenizer
    model, tokenizer = load_model_and_tokenizer(base_model_id, bnb_config)

    # Tokenize the training and validation datasets
    tokenized_train_dataset = dataset_train.map(generate_and_tokenize_prompt)
    tokenized_val_dataset = dataset_val.map(generate_and_tokenize_prompt)

    # Enable gradient checkpointing in the model and prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Print trainable parameters in the model
    print_trainable_parameters(model)

    # Configure LORA (LOng-Range dependencies in Recurrent Architectures) parameters using LoraConfig
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    # Apply LORA configuration to the model
    model = configure_lora(model, config)

    # Check if more than 1 GPU is available and update model properties accordingly
    if check_gpu_count():
        model.is_parallelizable = True
        model.model_parallel = True

    # Configure the Trainer for training the model
    trainer = configure_trainer(model, tokenized_train_dataset, tokenized_val_dataset, tokenizer)

    # Measure training time and print the elapsed time
    start_time = time.time()
    model.config.use_cache = False
    trainer.train()
    print("--- %s seconds ---" % (time.time() - start_time))



if __name__ == '__main__':
    # Train the language model
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file_source", default="val.source")
    parser.add_argument("--pred_file_target", default= "val.target")

    parser.add_argument("--train_file_source", default= "train.source")
    parser.add_argument("--train_file_target", default= "train.target")
    
    parser.add_argument("--base_model_name", default="mistral")

    args = parser.parse_args()

    # Define the base model ID for the Language Model (LLM)    
    if args.base_model_name = "mistral":
        base_model_id = "mistralai/Mistral-7B-v0.1"
    elif args.base_model_name = "starling":
        base_model_id = "berkeley-nest/Starling-LM-7B-alpha"
    elif args.base_model_name = "llama_7b:"
        base_model_id = "meta-llama/Llama-2-7b-hf"
    else : 
        base_model_id = "meta-llama/Llama-2-13b-hf"
    
    train_language_model(base_model_id, args.train_file_target, args.train_file_source, args.pred_file_target, args.pred_file_source)
