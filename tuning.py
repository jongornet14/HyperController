import torch
from transformers import GPT2Model, GPT2Tokenizer
from peft import PeftModel, get_peft_model, LoraConfig
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.nn as nn
import argparse
from collections import defaultdict
import time
import os 
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import hyper_framework
import tqdm 

parser = argparse.ArgumentParser()
# parser.add_argument("--method", type=str, default="Random") 
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--t_ready", type=int, default=int(400))
parser.add_argument("--folder", type=str, default='temp')
parser.add_argument("--method", type=str, default="Random") 
parser.add_argument("--home_dir", type=str, default='/home/jonathangornet/Documents/')

args = parser.parse_args()

torch.manual_seed(args.seed)

method = args.method

# Load the pre-trained GPT-2 model and tokenizer
model = GPT2Model.from_pretrained("gpt2-medium")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")


# Create a LoRA configuration
lora_config = LoraConfig(
    r=8,  # rank of the LoRA matrices
    lora_alpha=32,
    lora_dropout=0.05,
)


# Create a PeftModel instance with LoRA
peft_model = get_peft_model(model, lora_config)


# Load the SNLI dataset
dataset = load_dataset("snli")
# dataset = load_dataset("tatsu-lab/alpaca")
# dataset = load_dataset("roneneldan/TinyStories")

# Create a custom dataset class for the SNLI dataset
class SNLIDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        premise = self.dataset[idx]["premise"]
        hypothesis = self.dataset[idx]["hypothesis"]
        label = self.dataset[idx]["label"]

        if label == -1:
            return None

        encoding = self.tokenizer(premise, hypothesis, return_tensors="pt", max_length=512, truncation=True)

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label),
        }


# Define a custom collate function
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = torch.tensor([item["labels"] for item in batch])

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# Create a data loader for the SNLI dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = SNLIDataset(dataset["train"], tokenizer)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

#------------------------------------ Logs -----------------------------------------

logs = defaultdict(list)

t0 = time.time()
t1 = time.time()

logdir = "{}_{}_seed{}".format(
        'LLM',
        method,
        args.seed,
    )

if not os.path.exists(os.path.join(args.folder,logdir)):
    os.makedirs(os.path.join(args.folder,logdir))

writer = SummaryWriter(os.path.join(args.folder,logdir))

#------------------------------------ Scheduler ------------------------------------

hyperparameter_bounds = {
    "lr": [1e-5, 1e-3],
    "beta1": [0.5, 0.9],
    "beta2": [0.5, 0.999],
}

scheduler = hyper_framework.Scheduler(hyperparameter_bounds,args.t_ready,method)

iters = 0

print(len(data_loader.dataset)//64)

# Fine-tune the model with LoRA
peft_model.to(device)
optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()
for epoch in range(10):
    peft_model.train()
    for i, batch in enumerate(data_loader):
        inputs = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = peft_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.last_hidden_state[:, 0, :]
        loss = criterion(logits, inputs["labels"])
        loss.backward()
        optimizer.step()

        #------------------------------------ Hyperparams ------------------------------------
    
        logs["loss"].append(loss.item())
    
        logs["Reward"].append( - loss.item() )
        logs["reward"].append( - loss.item() )
        
        logs["lr"].append(optimizer.param_groups[0]["lr"])
        logs["beta1"].append(optimizer.param_groups[0]["betas"][0])
        logs["beta2"].append(optimizer.param_groups[0]["betas"][1])
        
        logs['Time'].append(t1-t0)
        logs['Trial'].append('hyperparam_trial')
        logs['iteration'].append(iters)
    
        # Optimizer
        t0 = time.time()
        config_dict = scheduler.step(logs,pd.DataFrame(logs))
        t1 = time.time()
        
        for g in optimizer.param_groups:
            g['lr']           = config_dict['lr']
            g['betas']        = (config_dict['beta1'],config_dict['beta2'])
    
        writer.add_scalar(os.path.join(args.folder,logdir,'performance/reward'), logs['reward'][-1], i)
        writer.add_scalar(os.path.join(args.folder,logdir, 'hyperparams/learning_rate'), logs["lr"][-1], i)
        writer.add_scalar(os.path.join(args.folder,logdir, 'hyperparams/beta1'), logs["beta1"][-1], i)
        writer.add_scalar(os.path.join(args.folder,logdir, 'hyperparams/beta2'), logs["beta2"][-1], i)

        iters += 1
        
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

savefilepath = os.path.join(args.folder,logdir,'logs.csv')

pd.DataFrame(logs).to_csv(savefilepath)