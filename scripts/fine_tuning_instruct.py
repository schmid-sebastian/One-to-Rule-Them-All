import os
import shutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from matplotlib import pyplot as plt
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from datasets import load_dataset, load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments


def apply_template(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    text = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        messages = [
            {"role": "system", "content": "You are a helpful assistant proficient in Process Mining"},
            {"role": "user", "content": instruction + input},
            {"role": "assistant", "content": output}
        ]
        text.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return {"text" : text}


if __name__ == "__main__":
    model, tokenizer = FastLanguageModel.from_pretrained(
        #model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length = 16384,
        dtype = None,
        load_in_4bit = True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
        use_rslora=True,
        use_gradient_checkpointing="unsloth"
    )
    
    print("Loading data...")
    dataset = load_dataset("json", data_dir='../data/instruction/train/')
    dataset = dataset.map(apply_template, batched = True, num_proc=16)
    #print("Loading data from disk...")
    #dataset = load_from_disk("../data/hf_datasets/")
    split_dataset = dataset['train'].train_test_split(test_size=0.05)
    train_dataset = split_dataset['train']
    eval_dataset  = split_dataset['test']
    
    trainer = SFTTrainer(
        model = model, # The model with LoRA adapters
        tokenizer = tokenizer, # The tokenizer of the model
        train_dataset = train_dataset, # The dataset to use for training
        eval_dataset  = eval_dataset,
        dataset_text_field = "text", # The field in the dataset that contains the structured data
        max_seq_length = 16384, # Max length of input sequence that the model can process
        dataset_num_proc = None, # CPU cores to use for loading and processing the data
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2, # Training Batch size per GPU
            per_device_eval_batch_size = 1, # Eval batch size per GPU
            gradient_accumulation_steps = 8, # Step size of gradient accumulation
            warmup_steps = 25,
            num_train_epochs = 1, # Set this for 1 full training run.
            # max_steps = 600, # Maximum steps of training
            learning_rate = 2e-4, # Initial learning rate
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            eval_strategy = "steps",
            eval_steps = 75,
            optim = "adamw_8bit", # The optimizer that will be used for updating the weights
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "../models/backup checkpoints/",
            report_to = "none", # Use this for WandB etc
        ),
    )
    

    
    trainer_stats = trainer.train()
    
    # Define the source folder (in the current working directory)
    source = "OTRTA_v7_instruct_base_new_system/"

    # Define the destination folder relative to the current working directory
    destination = "../models/"
    
    # Save the fine-tuned model to the specified directory
    trainer.save_model(source)

    # Also save the tokenizer, if needed
    tokenizer.save_pretrained(source)
    
    # Move the folder
    shutil.move(source, destination)

    print("Folder moved successfully to", os.path.join(destination, source))
        
    # trainer.state.log_history contains logged metrics (including "eval_loss")
    eval_losses = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
    if eval_losses:
        plt.plot(eval_losses)
        plt.xlabel("Evaluation Step")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss Over Training")
        plt.savefig(os.path.join(destination, source, "validation_loss.png"))  # Saves the plot to a file
        print("Validation loss plot saved as 'validation_loss.png'")
    else:
        print("No evaluation loss logs were found.")