import sys
import re
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
import transformers
from transformers import AlbertTokenizer, AlbertModel, DistilBertTokenizer, DistilBertModel, RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import wandb
device = 'cuda' if cuda.is_available() else 'cpu'

"""
This script provides the training loop for  multi-task learning where
Task 1 and Task 2 are trained separately. This ouputs model.bin files and
predictions for further analysis.

"""

tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
class NetMultiTask(torch.nn.Module):
    def __init__(self):
        super(NetMultiTask, self).__init__()
        self.net = RobertaModel.from_pretrained("roberta-base")
        
        self.pre_classifier1 = torch.nn.Linear(768, 768)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.classifier1 = torch.nn.Linear(768, 2)
        
        self.pre_classifier2 = torch.nn.Linear(768, 768)
        self.dropout2 = torch.nn.Dropout(0.3)
        self.classifier2 = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        output_1 = self.net(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
      
        pooler1 = self.pre_classifier1(pooler)
        pooler1 = torch.nn.ReLU()(pooler1)
        pooler1 = self.dropout1(pooler1)
        output1 = self.classifier1(pooler1)

        pooler2 = self.pre_classifier2(pooler)
        pooler2 = torch.nn.ReLU()(pooler2)
        pooler2 = self.dropout2(pooler2)
        output2 = self.classifier2(pooler2)
        
        return output1, output2

# Mapping the text sentiment labels
def map_sentiment(x):
    if x == "negative":
        return 1
    elif x =="neutral":
        return 0
    elif x =="positive":
        return 2
    else:
        return None
    
# Custom PyTorch Datasets for Disaster and Sentiment Tweets
class DisasterData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.target
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding = 'max_length',
            return_token_type_ids=False
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        #token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            #'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }
   
class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.target
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding = 'max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        #token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            #'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)


def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

# Training loop for multi-task learning to take into account the two outputs
def train(model, training_loader, testing_loader, mode):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0

    tr_loss_val = 0
    n_correct_val = 0
    nb_tr_steps_val = 0
    nb_tr_examples_val = 0

    model.train()

    for _,data in enumerate(tqdm(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        #token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        output1, output2 = model(ids, mask)

        if mode == 1:
            output = output1
        elif mode == 2:
            output = output2
        else:
            assert False, 'Bad Task ID passed'


        loss = loss_function(output, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(output.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        loss_step = tr_loss/nb_tr_steps
        accu_step = (n_correct*100)/nb_tr_examples

        train_metrics = {"train_loss": loss_step,
                         "train_accuracy": accu_step}
        
        wandb.log({**train_metrics})

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(testing_loader):
            ids_val = data['ids'].to(device, dtype = torch.long)
            mask_val = data['mask'].to(device, dtype = torch.long)
            #token_type_ids_val = data['token_type_ids'].to(device, dtype = torch.long)
            targets_val = data['targets'].to(device, dtype = torch.long)

            output1_val, output2_val = model(ids_val, mask_val)

            if mode == 1:
                output_val = output1_val
            elif mode == 2:
                output_val = output2_val
            else:
                assert False, 'Bad Task ID passed'

            loss_val = loss_function(output_val, targets_val)
            tr_loss_val += loss_val.item()
            big_val_val, big_idx_val = torch.max(output_val.data, dim=1)
            n_correct_val += calcuate_accuracy(big_idx_val, targets_val)

            nb_tr_steps_val += 1
            nb_tr_examples_val += targets_val.size(0)
    
            loss_step_val = tr_loss_val/nb_tr_steps_val
            accu_step_val = (n_correct_val*100)/nb_tr_examples_val

            val_metrics = {"val_loss": loss_step_val,
                "val_accuracy": accu_step_val}

        wandb.log({**val_metrics})

    return

def valid(model, testing_loader, mode):
    model.eval()
    #n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    tr_loss=0
    predicts = []

    with torch.no_grad():
        for _, data in enumerate(tqdm(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            #token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)

            output1, output2 = model(ids, mask)

            if mode == 1:
                output = output1
            elif mode == 2:
                output = output2
            else:
                assert False, 'Bad Task ID passed'

            loss = loss_function(output, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(output.data, dim=1)
            
            for i in range(targets.size(0)):
                predicts.append({
                    "predict": big_idx[i].item(),
                    "target": targets[i].item()
                })
    df = pd.DataFrame(predicts)
    return df

dir = sys.argv[1]

d_train = pd.read_csv(f"{dir}/train.csv")
d_test = pd.read_csv(f"{dir}/test.csv")
s_train = pd.read_csv(f"{dir}/tweets.csv")

s_train.drop(s_train[s_train["textID"]=="fdb77c3752"].index, inplace=True)

with open(f"{dir}/wandb_key.txt", "r") as f:
    wandb_key = f.read()

wandb.login(key=wandb_key)
# Drop duplicates
# d_train.drop_duplicates(subset=['text'], inplace=True)
# s_train.drop_duplicates(subset=['text'], inplace=True)

d_train['id'] = 1
s_train['id'] = 2
d_train.reset_index(inplace=True)
s_train.reset_index(inplace=True)
s_train_text = s_train[['text','id','index']].copy()
d_train_text = d_train[['text','id','index']].copy()

s_train['sentiment'] = s_train.apply(lambda x: map_sentiment(x.sentiment), axis=1)
s_train.rename(columns={'sentiment':'target'}, inplace=True)

d_train_select =  d_train[['text','target']].copy()
s_train_select = s_train[['text','target']].copy()

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 32

# MAX_LEN = 512
# TRAIN_BATCH_SIZE = 32
# VALID_BATCH_SIZE = 32

# Create train-validate spilts
d_train_data, d_val_data = train_test_split(d_train_select, test_size=0.2, stratify=d_train_select['target'],
                                 random_state=2023)

s_train_data, s_val_data= train_test_split(s_train_select, test_size=0.2, stratify=s_train_select['target'],
                                 random_state=2023)

d_train_data.reset_index(inplace=True,drop = True)
d_val_data.reset_index(inplace=True, drop = True)
s_train_data.reset_index(inplace=True, drop = True)
s_val_data.reset_index(inplace=True,  drop = True)

# Create D1 and D2 Datasets
d1_train_set= DisasterData(d_train_data, tokenizer, MAX_LEN)
d1_val_set = DisasterData(d_val_data, tokenizer, MAX_LEN)

d2_train_set= SentimentData(s_train_data, tokenizer, MAX_LEN)
d2_val_set = SentimentData(s_val_data, tokenizer, MAX_LEN)


# Create D1 and D2 dataloaders
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

d1_train_loader = DataLoader(d1_train_set, **train_params)
d1_val_loader = DataLoader(d1_val_set, **test_params)

d2_train_loader = DataLoader(d2_train_set, **train_params)
d2_val_loader = DataLoader(d2_val_set, **test_params)

LEARNING_RATE = 1e-05
# epochs = 30
# Predict on first task
net1 = NetMultiTask()
net1.to(device)
#for training only the classification layer

# for param in net1.net.parameters():
#     param.requires_grad = False

EPOCHS = 2
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = net1.parameters(), lr = LEARNING_RATE)

wandb.init(
        project="bt5151_multitask",
        group='task1',
        config={
            "epochs": EPOCHS,
            "batch_size": TRAIN_BATCH_SIZE,
            "lr": LEARNING_RATE,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "max_length": MAX_LEN
            })
config = wandb.config
for epoch in range(EPOCHS):
    train(net1, d1_train_loader, d1_val_loader, mode = 1)
    print('Finished training')


predicts_d1  = valid(net1, d1_val_loader, mode = 1)
d1_f1 = f1_score(predicts_d1.target, predicts_d1.predict,  average='weighted')
d1_accuracy = accuracy_score(predicts_d1.target, predicts_d1.predict)

eval_metrics_d1 = {"val_f1_score": d1_f1, "val_fin_accuracy": d1_accuracy}
wandb.log({**eval_metrics_d1})
wandb.finish()
print(classification_report(predicts_d1.target, predicts_d1.predict))

# Predict on second task
net2 = NetMultiTask()
net2.to(device)

# for param in net2.net.parameters():
#     param.requires_grad = False

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = net2.parameters(), lr = LEARNING_RATE)
EPOCHS_T2 = 2

wandb.init(
        project="bt5151_multitask",
        group='task2',
        config={
            "epochs": EPOCHS_T2,
            "batch_size": TRAIN_BATCH_SIZE,
            "lr": LEARNING_RATE,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "max_length": MAX_LEN
            })
config = wandb.config
for epoch in range(EPOCHS_T2):
    train(net2, d2_train_loader, d2_val_loader, mode = 2)
print('Finished training')

predicts_d2  = valid(net2, d2_val_loader, mode = 2)
d2_f1 = f1_score(predicts_d2.target, predicts_d2.predict, average='weighted')
d2_accuracy = accuracy_score(predicts_d2.target, predicts_d2.predict)
eval_metrics_d2 = {"val_f1_score": d2_f1, "val_fin_accuracy": d2_accuracy}
wandb.log({**eval_metrics_d2})
wandb.finish()

print(classification_report(predicts_d2.target, predicts_d2.predict))

net1_bin = f"{dir}/models/net1.bin"
net2_bin = f"{dir}/models/net2.bin"

predicts_d1.to_csv(f"{dir}/predicts/predicts_d1.csv")
predicts_d2.to_csv(f"{dir}/predicts/predicts_d2.csv")
torch.save(net1, net1_bin)
torch.save(net2, net2_bin)

print(f"D1 F1: {d1_f1}\n"
      f"D2 F2: {d2_f1}\n"
      f"D1 Accuracy: {d1_accuracy}\n"
      f"D2 Accuracy: {d2_accuracy}\n")