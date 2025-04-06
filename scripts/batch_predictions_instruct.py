import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import unsloth
from unsloth import FastLanguageModel
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import random
import torch

def generate_prompt(example):
    instructions = example["instruction"]
    inputs = example["input"]
    messages = [
        {"role": "system", "content": "You are a helpful assistant proficient in Process Mining"},
        {"role": "user", "content": instructions + inputs},
    ]
    
    # Get the prompt text without tokenizing.
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Get raw text output.
        add_generation_prompt=True, # doesnt matter for llama models
    )
    
    return prompt_text

if __name__ == "__main__":
    local_model_path = "../models/OTRTA_v7_instruct_base_new_system/"
    model_name = os.path.basename(os.path.normpath(local_model_path))
    output_filename = f"../data/instruction/eval_predictions/predictions_{model_name}.json"
    
    # Define the directory for individual output files.
    individual_output_dir = "../data/instruction/eval_predictions/individual"
    os.makedirs(individual_output_dir, exist_ok=True)
    existing_evals = os.listdir(individual_output_dir)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=local_model_path,
        max_seq_length=16384,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.for_inference(model)

    TEST_PATH = '../data/instruction/test/'
    test_files = []
    test_jsons = os.listdir(TEST_PATH)
    #sample_size = 20
    #print(f"Sampling {sample_size}/{len(test_jsons)} test samples...")
    #test_jsons = random.sample(test_jsons, sample_size)
    output_data = {}

    print("Processing test JSON files...")
    for file in tqdm(test_jsons):
        if file.endswith(".json"):
            if file in existing_evals:
                continue
            else:
                with open(os.path.join(TEST_PATH, file), encoding="utf-8") as f:
                    d = json.load(f)
                sample = d[0]
                sample["file_name"] = file
                test_files.append(sample)

    batch_size = 2
    print(f"Generating LLM responses in batches of {batch_size}...")

    for i in tqdm(range(0, len(test_files), batch_size)):
        batch_samples = test_files[i:i + batch_size]
        
        # Generate prompts for each sample in the current batch.
        prompts = [generate_prompt(sample) for sample in batch_samples]
        
        # Compute maximum tokenized length in the batch.
        max_length = max(len(tokenizer(prompt)["input_ids"]) for prompt in prompts)
        print(max_length)
        
        if max_length > 6000:
            print("Skipping current batch since batch size is", max_length)
            continue
        
        # Tokenize with consistent length.
        inputs = tokenizer(prompts, padding='longest', return_tensors="pt").to("cuda")
        
        # Generate outputs for the batch.
        outputs = model.generate(**inputs, max_new_tokens=int(max_length*1.25), use_cache=True)
        
        # Remove the input from the generated output
        input_length = inputs['input_ids'].shape[1]
        answers = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
        
        # Store predictions for each sample.
        for sample, y_pred in zip(batch_samples, answers):
            sample["predicted_output"] = y_pred
            output_data[sample["file_name"]] = sample
            
            # Save individual file.
            indiv_output_filename = os.path.join(individual_output_dir, sample["file_name"])
            with open(indiv_output_filename, "w", encoding="utf-8") as f:
                json.dump(sample, f, indent=4)
        
    
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(f"Predictions saved to {output_filename}")
