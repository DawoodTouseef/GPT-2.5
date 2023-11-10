from transformers import GPT2LMHeadModel, GPT2Tokenizer
from GPT2 import ChatData
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch
def train(chatData, model, optim):
    epochs = 3
    for i in tqdm.tqdm(range(epochs)):
        print("epoch " + str(i))
        for X, a in chatData:
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            loss.backward()
            optim.step()
        torch.save(model.state_dict(), "model_state.pt")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]","bos_token": "[INST]","eos_token": "[/INST]"})
tokenizer.add_tokens(["<bot>:"])
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)
# print(tokenizer.decode(model.generate(**tokenizer("hey i was good at basketball but ",
#                          return_tensors="pt"))[0]))
chatData = ChatData(tokenizer)
chatData =  DataLoader(chatData, batch_size=64)
model.train()
optim = Adam(model.parameters(), lr=1e-5)
print("training .... ")
train(chatData, model, optim)