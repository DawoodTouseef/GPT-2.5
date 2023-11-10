import os.path
import torch.cuda
from transformers import GPT2Tokenizer, GPT2Model,TrainingArguments,Trainer
from datasets import Dataset
from datasets import load_dataset

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model=GPT2Model.from_pretrained("gpt2")
truthful_qa = load_dataset('truthful_qa',"generation")
alpaca_gpt4 = load_dataset("vicgalle/alpaca-gpt4")

# Extract relevant data from the datasets
truthful_qa_train = truthful_qa['validation']
alpaca_gpt4_train_instruction = alpaca_gpt4['train']['instruction']
alpaca_gpt4_train_text = alpaca_gpt4['train']['text']
alpaca_gpt4_train_output = alpaca_gpt4['train']['output']

# Combine data into training examples
train_dataset = []

# Add truthful_qa data
for i in range(len(truthful_qa_train)):
    train_dataset.append({
        "input_text": truthful_qa_train[i]['question'],
        "output_text": truthful_qa_train[i]['best_answer'],
    })
# Add alpaca_gpt4 data
for i in range(len(alpaca_gpt4_train_instruction)):
    train_dataset.append({
        "input_text": alpaca_gpt4_train_instruction[i],
        "output_text": alpaca_gpt4_train_output[i],
    })

for i in range(len(alpaca_gpt4_train_text)):
    train_dataset.append({
        "input_text": alpaca_gpt4_train_text[i],
        "output_text": alpaca_gpt4_train_output[i],
    })
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
def tokenize_function(example):
    input_text = f'''
            [INST]
            <<SYS>>Your Helpful,loyal and respectful Assistant.Your Name is JARVIS(Just a Rather Very Intelligent System)<</SYS>>
            {example['input_text']}[/INST]
            {example['output_text']}'''
    return input_text

train_datasets=[tokenize_function(x) for x in train_dataset]
dataset=Dataset.from_dict({"text":train_datasets})
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
print(tokenized_datasets)
print("Training the GPT-2 model ......")
output="gpt2-1B"
output_dir=False
if os.path.exists(f"./{output}"):
    output_dir=True
if torch.cuda.is_available():
    cpu=False
else:
    cpu=True
trainer_arg=TrainingArguments(
    output_dir=output,
    overwrite_output_dir=output_dir,
    num_train_epochs=3,
    learning_rate=1e-5,
    use_cpu=cpu,
)
trainer=Trainer(
    args=trainer_arg,
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets
)
trainer.train()