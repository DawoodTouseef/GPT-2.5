from torch.utils.data import Dataset
from datasets import load_dataset

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

def tokenize_function(example):
    input_text = f'''
            [INST]
            <<SYS>>Your Helpful,loyal and respectful Assistant.Your Name is JARVIS(Just a Rather Very Intelligent System)<</SYS>>
            {example['input_text']}[/INST]
            {example['output_text']}'''
    return input_text
train_datasets=[tokenize_function(x) for x in train_dataset]

class ChatData(Dataset):
    def __init__(self, tokenizer):

        self.X = train_datasets

        self.X = self.X[:5000]

        print(self.X[0])
        print(len(self.X))

        self.X_encoded = tokenizer(self.X, max_length=40, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])

