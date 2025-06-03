#Import Libraries
import torch
import os
from transformers import AutoModelForCausalLM, AutoProcessor
from torch.utils.data import Dataset, Dataloader
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm 
from transformers import AdamW, get_scheduler
import torch.multiprocessing as mp

#load the model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft",trust_remote_code=True,revision='refs/pr/6').to(device) 
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision='refs/pr/6')

for param in model.vision_tower.parameters():
    param.is_trainable = False

#load the dataset
dataset = load_dataset("json", data_files="finetuning_smolvlm/passengers_bus_vlm_dataset_modified.json")
print(dataset)


#split the dataset
dataset = load_dataset("json", data_files="finetuning_smolvlm/passengers_bus_vlm_dataset_modified.json")

split_ds = dataset["train"].train_test_split(test_size=0.2, seed=42)  # 80% train, 20% validation
validation_test_split = split_ds["test"].train_test_split(test_size=0.5, seed=42)

train = split_ds["train"]
val = validation_test_split["train"]
test = validation_test_split["test"]
print(f"Train size: {len(train)}, Validation size: {len(val)}, Test size: {len(test)}")


#Create the dataclass
class VQADataset(Dataset): 

    def __init__(self, data): 
        self.data = data
        
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, idx):
        example = self.data[idx]
        question = "<VQA>" + example['question'] 
        first_answer = example['answer']
        image = Image.open(example['image'])
        image = image.convert("RGB")
        return question, first_answer, image
    
    
#the collate function
def collate_fn(batch): 
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers 

train_dataset = VQADataset(train)
val_dataset = VQADataset(val) 
batch_size = 1
num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)


#define the train function
def train():
    epochs = 7
    optimizer = AdamW(model.parameters(), lr=1e-6)
    num_training_steps = epochs * len(train_loader)

    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, 
                                num_warmup_steps=0, num_training_steps=num_training_steps,)
    for epoch in range(epochs): 
        model.train() 
        train_loss = 0
        i = -1
        for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            i += 1
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"] 
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                inputs, answers = batch
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

        print(val_loss / len(val_loader))
        
        # Save model checkpoint
        output_dir = f"./model_checkpoints/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        
        
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train()


