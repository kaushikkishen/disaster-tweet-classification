import sys
import re
import math
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

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.net(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
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

def map_sentiment(x):
    if x == "negative":
        return 1
    elif x =="neutral":
        return 0
    elif x =="positive":
        return 2
    else:
        return None
    
class DataCombined(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.target = self.data.target
        self.sentiment = self.data.sentiment
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
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': (self.target[index], self.sentiment[index])
        }

def check_null(x):
    if math.isnan(x) == True:
        print(f"null detected")
        return 0
    else:
        return x

def null_tensor(tensor):
    if torch.isnan(tensor) == True:
        return torch.tensor(0)
    else:
        return tensor


def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def train_hydra(model, epoch, training_loader, testing_loader, lambda1, lambda2):
    d1_tr_loss = 0
    d1_n_correct = 0
    d1_nb_tr_steps = 0
    d1_nb_tr_examples = 0
    d2_tr_loss = 0
    d2_n_correct = 0
    d2_nb_tr_steps = 0
    d2_nb_tr_examples = 0
    total_tr_loss = 0
    
    d1_val_loss = 0
    d1_val_n_correct = 0
    d1_nb_val_steps = 0
    d1_nb_val_examples = 0
    
    d2_val_loss = 0
    d2_val_n_correct = 0
    d2_nb_val_steps = 0
    d2_nb_val_examples = 0
    total_val_loss = 0

    model.train()
    for loop, data in enumerate(tqdm(training_loader, 0)):
        d1_ids = data['ids'][~np.isnan(data['labels'][0].numpy()) == True].to(device, dtype = torch.long)
        d1_mask = data['mask'][~np.isnan(data['labels'][0].numpy()) == True].to(device, dtype = torch.long)
        d1_token_type_ids = data['token_type_ids'][~np.isnan(data['labels'][0].numpy()) == True].to(device, dtype = torch.long)
        d1_targets = data['labels'][0][~np.isnan(data['labels'][0].numpy()) == True].long().to(device, dtype = torch.long)

        d2_ids = data['ids'][~np.isnan(data['labels'][1].numpy()) == True].to(device, dtype = torch.long)
        d2_mask = data['mask'][~np.isnan(data['labels'][1].numpy()) == True].to(device, dtype = torch.long)
        d2_token_type_ids = data['token_type_ids'][~np.isnan(data['labels'][1].numpy()) == True].to(device, dtype = torch.long)
        d2_sentiment = data['labels'][1][~np.isnan(data['labels'][1].numpy()) == True].long().to(device, dtype = torch.long)


        output1, _ = model(d1_ids, d1_mask, d1_token_type_ids)
        _, output2 = model(d2_ids, d2_mask, d2_token_type_ids)

        loss1 = loss_function(output1, d1_targets)
        loss1 = null_tensor(loss1)
        d1_tr_loss += check_null(lambda1*loss1.item())

        loss2 = loss_function(output2, d2_sentiment)
        loss2 = null_tensor(loss2)
        d2_tr_loss += check_null(lambda2*loss2.item())

        total_loss = (lambda1*loss1) + (lambda2*loss2)
        total_tr_loss += total_loss.item()
        

        big_val_d1, big_idx_d1 = torch.max(output1.data, dim=1)
        d1_n_correct += calcuate_accuracy(big_idx_d1, d1_targets)

        big_val_d2, big_idx_d2 = torch.max(output2.data, dim=1)
        d2_n_correct += calcuate_accuracy(big_idx_d2, d2_sentiment)

        d1_nb_tr_steps += 1
        d1_nb_tr_examples += d1_targets.size(0)

        d2_nb_tr_steps += 1
        d2_nb_tr_examples += d2_sentiment.size(0)
        

        d1_loss_step = d1_tr_loss/d1_nb_tr_steps
        d1_accu_step = (d1_n_correct*100)/d1_nb_tr_examples


        d2_loss_step = d2_tr_loss/d2_nb_tr_steps
        d2_accu_step = (d2_n_correct*100)/d2_nb_tr_examples

        tr_loss_step = d1_loss_step + d2_loss_step
        
        optimizer.zero_grad()
        total_loss.backward()
        # # When using GPU
        optimizer.step()

        train_metrics = {"d1_train_loss": d1_loss_step,
            "d1_train_accuracy": d1_accu_step,
            "d2_train_loss": d2_loss_step,
            "d2_train_accuracy": d2_accu_step,
            "total_train_loss": tr_loss_step}
        
        wandb.log({**train_metrics})

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(tqdm(testing_loader, 0)):
            d1_ids_val = data['ids'][~np.isnan(data['labels'][0].numpy()) == True].to(device, dtype = torch.long)
            d1_mask_val = data['mask'][~np.isnan(data['labels'][0].numpy()) == True].to(device, dtype = torch.long)
            d1_token_type_ids_val = data['token_type_ids'][~np.isnan(data['labels'][0].numpy()) == True].to(device, dtype = torch.long)
            d1_targets_val = data['labels'][0][~np.isnan(data['labels'][0].numpy()) == True].long().to(device, dtype = torch.long)

            d2_ids_val = data['ids'][~np.isnan(data['labels'][1].numpy()) == True].to(device, dtype = torch.long)
            d2_mask_val = data['mask'][~np.isnan(data['labels'][1].numpy()) == True].to(device, dtype = torch.long)
            d2_token_type_ids_val = data['token_type_ids'][~np.isnan(data['labels'][1].numpy()) == True].to(device, dtype = torch.long)
            d2_sentiment_val = data['labels'][1][~np.isnan(data['labels'][1].numpy()) == True].long().to(device, dtype = torch.long)

            output1_val, _ = model(d1_ids_val, d1_mask_val, d1_token_type_ids_val)
            _, output2_val = model(d2_ids_val, d2_mask_val, d2_token_type_ids_val)

            loss1_val = loss_function(output1_val, d1_targets_val)
            loss1_val = null_tensor(loss1_val)
            d1_val_loss += check_null(lambda1*loss1_val.item())

            loss2_val = loss_function(output2_val, d2_sentiment_val)
            loss2_val = null_tensor(loss2_val)
            d2_val_loss += check_null(lambda2*loss2_val.item())
            
            total_loss_val = (lambda1*loss1_val) + (lambda2*loss2_val)
            total_val_loss += total_loss_val.item()
            
            big_val_d1, big_idx_d1_val = torch.max(output1_val.data, dim=1)
            d1_val_n_correct += calcuate_accuracy(big_idx_d1_val, d1_targets_val)

            big_val_d2, big_idx_d2_val = torch.max(output2_val.data, dim=1)
            d2_val_n_correct += calcuate_accuracy(big_idx_d2_val, d2_sentiment_val)

            d1_nb_val_steps += 1
            d1_nb_val_examples += d1_targets_val.size(0)

            d2_nb_val_steps += 1
            d2_nb_val_examples += d2_sentiment_val.size(0)
            
            try:
                d1_loss_step_val = d1_val_loss/d1_nb_val_steps
            except ZeroDivisionError:
                d1_loss_step_val = 0
            try:
                d1_accu_step_val = (d1_val_n_correct*100)/d1_nb_val_examples
            except ZeroDivisionError:
                d1_accu_step_val = 0


            try:
                d2_loss_step_val = d2_val_loss/d2_nb_val_steps
            except ZeroDivisionError:
                d2_loss_step_val = 0
            
            try:
                d2_accu_step_val = (d2_val_n_correct*100)/d2_nb_val_examples
            except ZeroDivisionError:
                d2_accu_step_val = 0

            tr_loss_step_val = d1_loss_step_val + d2_loss_step_val

            test_metrics = {"d1_test_loss": d1_loss_step_val,
            "d1_test_accuracy": d1_accu_step_val,
            "d2_test_loss": d2_loss_step_val,
            "d2_test_accuracy": d2_accu_step_val,
            "total_test_loss": tr_loss_step_val}
            
    wandb.log({**test_metrics})
    # print(f"D1 Training Loss per 500 steps: {d1_loss_step}")
    # print(f"D1 Training Accuracy per 500 steps: {d1_accu_step}\n")

    # print(f"D2 Training Loss per 500 steps: {d2_loss_step}")
    # print(f"D2 Training Accuracy per 500 steps: {d2_accu_step}\n")

    # print(f"Total Training Loss per 500 steps: {tr_loss_step}\n")

    print(f'Total D1 Accuracy for Epoch {epoch}: {(d1_n_correct*100)/d1_nb_tr_examples}')
    print(f'Total D2 Accuracy for Epoch {epoch}: {(d2_n_correct*100)/d2_nb_tr_examples}')
    
    return

def valid_hydra(model, testing_loader):
    model.eval()
    d1_predicts = []
    d2_predicts = []

    with torch.no_grad():
        for _, data in enumerate(tqdm(testing_loader, 0)):
            d1_ids = data['ids'][~np.isnan(data['labels'][0].numpy()) == True].to(device, dtype = torch.long)
            d1_mask = data['mask'][~np.isnan(data['labels'][0].numpy()) == True].to(device, dtype = torch.long)
            d1_token_type_ids = data['token_type_ids'][~np.isnan(data['labels'][0].numpy()) == True].to(device, dtype = torch.long)
            d1_targets = data['labels'][0][~np.isnan(data['labels'][0].numpy()) == True].long().to(device, dtype = torch.long)

            d2_ids = data['ids'][~np.isnan(data['labels'][1].numpy()) == True].to(device, dtype = torch.long)
            d2_mask = data['mask'][~np.isnan(data['labels'][1].numpy()) == True].to(device, dtype = torch.long)
            d2_token_type_ids = data['token_type_ids'][~np.isnan(data['labels'][1].numpy()) == True].to(device, dtype = torch.long)
            d2_sentiment = data['labels'][1][~np.isnan(data['labels'][1].numpy()) == True].long().to(device, dtype = torch.long)

            output1, _ = model(d1_ids, d1_mask, d1_token_type_ids)
            _, output2 = model(d2_ids, d2_mask, d2_token_type_ids)
            
            big_val_d1, big_idx_d1 = torch.max(output1.data, dim=1)
            big_val_d2, big_idx_d2 = torch.max(output2.data, dim=1)
            
            for i in range(d1_targets.size(0)):
                d1_predicts.append({
                    "predict": big_idx_d1[i].item(),
                    "target": d1_targets[i].item()
                })

            for i in range(d2_sentiment.size(0)):
                d2_predicts.append({
                    "predict": big_idx_d2[i].item(),
                    "target": d2_sentiment[i].item()
                })

    d1_df = pd.DataFrame(d1_predicts)
    d2_df = pd.DataFrame(d2_predicts)

    return d1_df, d2_df

dir = sys.argv[1]

d_train = pd.read_csv(f"{dir}/train.csv")
d_test = pd.read_csv(f"{dir}/test.csv")
s_train = pd.read_csv(f"{dir}/tweets.csv")

with open(f"{dir}/wandb_key.txt", "r") as f:
    wandb_key = f.read()

wandb.login(key=wandb_key)

s_train.drop(s_train[s_train["textID"]=="fdb77c3752"].index, inplace=True)

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

s_train_concat = s_train_data.rename(columns={"target":"sentiment"}).copy()
s_val_concat = s_val_data.rename(columns={"target":"sentiment"}).copy()

sd_train_data = pd.concat([d_train_data, s_train_concat], ignore_index=True)
sd_val_data = pd.concat([d_val_data, s_val_concat], ignore_index=True)

MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
LEARNING_RATE = 1e-05

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

sd_train_dataset = DataCombined(sd_train_data, tokenizer=tokenizer, max_len=MAX_LEN)
sd_val_dataset = DataCombined(sd_val_data, tokenizer=tokenizer, max_len=MAX_LEN)

sd_train_loader = DataLoader(sd_train_dataset, **train_params)
sd_val_loader = DataLoader(sd_val_dataset, **test_params)

net_hydra = NetMultiTask()
net_hydra.to(device)
EPOCHS = 2
loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params = net_hydra.parameters(), lr = LEARNING_RATE)
LAMBDA1 = 0.6916868733242154
LAMBDA2 = 0.6862697005271912
wandb.init(
        project="bt5151_hydra",
        group ="fix_loss",
        config={
            "epochs": EPOCHS,
            "batch_size": TRAIN_BATCH_SIZE,
            "lr": LEARNING_RATE,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "max_length": MAX_LEN,
            "lambda1": LAMBDA1,
            "lambda2": LAMBDA2,
            })

# Copy your config 
config = wandb.config
for epoch in range(EPOCHS):
    train_hydra(net_hydra, epoch, sd_train_loader, sd_val_loader, 
                lambda1 = LAMBDA1, lambda2 = LAMBDA2)
print('Finished training')

d1_predict, d2_predict  = valid_hydra(net_hydra, sd_val_loader)
fin_metrics = {"f1_score": f1_score(d1_predict.target, d1_predict.predict, average='weighted'),
               "d2_f1": f1_score(d2_predict.target, d2_predict.predict, average='weighted'),
               "d1_fin_accuracy": accuracy_score(d1_predict.target, d1_predict.predict),
               "d2_fin_accuracy": accuracy_score(d2_predict.target, d2_predict.predict)}
wandb.log({**fin_metrics})
wandb.finish()
try:
    net_bin = f"{dir}/models/net_hydra.bin"
    torch.save(net_hydra, net_bin)
except Exception as e: 
    print(e)

try:
    d1_predict.to_csv(f"{dir}/predicts/d1_predict.csv")
    d2_predict.to_csv(f"{dir}/predicts/d2_predict.csv")
except Exception as e: 
    print(e)

try:
    print(classification_report(d1_predict.target, d1_predict.predict))
    print(classification_report(d2_predict.target, d2_predict.predict))
    
    print(f"D1 F1: {f1_score(d1_predict.target, d1_predict.predict, average='weighted')}\n"
        f"D2 F2: {f1_score(d2_predict.target, d2_predict.predict, average='weighted')}\n"
        f"D1 Accuracy: {accuracy_score(d1_predict.target, d1_predict.predict)}\n"
        f"D2 Accuracy: {accuracy_score(d2_predict.target, d2_predict.predict)}\n")
except Exception as e: 
    print(e)
else:
    print("Success")

