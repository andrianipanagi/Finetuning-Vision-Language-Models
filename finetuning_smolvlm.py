#import libraries
import os
import json
import transformers
import datasets
import accelerate
import peft
import torch
import torchvision
import bitsandbytes

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

#Load model
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration

#for visible devices to be only 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = 0
USE_LORA = False
USE_QLORA = True
SMOL = True

model_id = "HuggingFaceTB/SmolVLM-Base" if SMOL else "HuggingFaceTB/SmolVLM-500M-Instruct" #"HuggingFaceM4/Idefics3-8B-Llama3"
processor = AutoProcessor.from_pretrained(model_id)

if USE_QLORA or USE_LORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian"
    )
    lora_config.inference_mode = False
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, # use 4-bit precision model loading
            bnb_4bit_use_double_quant=True, # apply nested quantization
            bnb_4bit_quant_type="nf4", # quantization type
            bnb_4bit_compute_dtype=torch.bfloat16 # compute dtype
        )

    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config if USE_QLORA else None,
        _attn_implementation="eager", # "flash_attention_2"
        device_map="cuda:0"
    )
    model.add_adapter(lora_config)
    model.enable_adapters()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print(model.get_nb_trainable_parameters())
else:
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        _attn_implementation= "eager", #"flash_attention_2",
    ).to(DEVICE)

    # if you'd like to only fine-tune LLM
    for param in model.model.vision_model.parameters():
        param.requires_grad = False
        
        
        
#Load the data
from datasets import load_dataset
dataset = load_dataset("json", data_files="passengers_bus_vlm_dataset_modified.json")

#split the dataset
split_ds = dataset["train"].train_test_split(test_size=0.2, seed=42)  # 80% train, 20% validation
validation_test_split = split_ds["test"].train_test_split(test_size=0.5, seed=42)

train = split_ds["train"]
val = validation_test_split["train"]
test = validation_test_split["test"]


image_token_id = processor.tokenizer.additional_special_tokens_ids[processor.tokenizer.additional_special_tokens.index("<image>")]

def collate_fn(examples):
  texts = []
  images = []
  for example in examples:
      image_path = example["image"]
      image = Image.open(image_path)
      if image.mode != 'RGB':
        image = image.convert('RGB')
      question = example["question"]
      answer = example["answer"]
      messages = [
          {"role": "system", "content": "You are a Vision Language Model specialized in extracting data from images.Your task is to analyze the provided image of the inside of a bus and extract the relevant information."},
          {
              "role": "user",
              "content": [
                  {"type": "image"},
                  {"type": "text", "text": question}
              ]
          },
          {
              "role": "assistant",
              "content": [
                  {"type": "text", "text": answer}
              ]
          }
      ]
      text = processor.apply_chat_template(messages, add_generation_prompt=False)
      texts.append(text.strip())
      images.append([image])

  batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
  labels = batch["input_ids"].clone()
  labels[labels == processor.tokenizer.pad_token_id] = -100
  labels[labels == image_token_id] = -100
  batch["labels"] = labels

  return batch


#Finetuning the model
from transformers import Trainer, TrainingArguments

model_name = model_id.split("/")[-1]

new_model = "SmolVLM-Base-vqav2"

training_args = TrainingArguments(
    num_train_epochs=4, #4 epochs for not overfitting
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    learning_rate=1e-5,
    weight_decay=0.0,
    logging_steps=10,
    save_strategy="epoch",
    #save_steps=25,
    save_total_limit=1,
    optim="paged_adamw_8bit",  # For 8-bit, else use "adamw_hf"
    fp16=True,  
    output_dir=f"./{model_name}-vqav2",
    hub_model_id=f"{model_name}-vqav2",
    report_to="wandb",
    run_name="SmolVLM-finetuning",
    remove_unused_columns=False,
    gradient_checkpointing=True
)

#use local rank in the training_args for only one visible device
#local_rank=-1,  # Ensures single-GPU training

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train,
    eval_dataset=val,
)

if __name__ == "__main__":
    # Train the model
    print("Starting training...")
    if USE_LORA or USE_QLORA:
        model.print_trainable_parameters()
    else:
        print("No LoRA or QLoRA, training all parameters.")
    
    trainer.train()
    
    # Save only the LoRA adapters
    trainer.model.save_pretrained(training_args.output_dir)


