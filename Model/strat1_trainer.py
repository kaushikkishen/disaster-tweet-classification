import sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import OrderedDict
import torch
import transformers
from transformers import AlbertTokenizer, AlbertModel, DistilBertTokenizer, DistilBertModel, RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
device = 'cuda' if cuda.is_available() else 'cpu'
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import evaluate

def map_sentiment(x):
    if x == "negative":
        return 0
    elif x =="neutral":
        return 1
    elif x =="positive":
        return 2
    else:
        return None

def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)

class HuggingData(Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels

  def __getitem__(self, idx):
      item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
      item['labels'] = torch.tensor(self.labels[idx])
      return item

  def __len__(self):
      return len(self.labels)
  
def tokenize_function(examples):
  return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)

dir = sys.argv[1]

d_train = pd.read_csv(f"{dir}/train.csv")
d_test = pd.read_csv(f"{dir}/test.csv")
s_train = pd.read_csv(f"{dir}/tweets.csv")

s_train.drop(s_train[s_train["textID"]=="fdb77c3752"].index, inplace=True)

# Remove URLs
# d_train['text'] = d_train.apply(lambda x: cleaning_URLs(x['text']),axis=1)
# #d_test['text'] = d_test.apply(lambda x: cleaning_URLs(x['text']),axis=1)
# s_train['text'] = s_train.apply(lambda x: cleaning_URLs(x['text']),axis=1)

# Drop duplicates
d_train.drop_duplicates(subset=['text'], inplace=True)
s_train.drop_duplicates(subset=['text'], inplace=True)

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

d_train_data, d_val_data, d_train_labels, d_val_labels = train_test_split(d_train_select['text'],d_train_select['target'], test_size=0.2, 
                                                                       stratify=d_train_select['target'], random_state=2023)


d_train_data.reset_index(inplace=True, drop=True)
d_val_data.reset_index(inplace=True, drop=True)
d_train_labels.reset_index(inplace=True, drop=True)
d_val_labels.reset_index(inplace=True, drop=True)

train_encodings = tokenizer(d_train_data.tolist(), truncation=True, padding="max_length", add_special_tokens=True, 
                            return_token_type_ids=True)
val_encodings = tokenizer(d_val_data.tolist(), truncation=True, padding="max_length", add_special_tokens=True, 
                            return_token_type_ids=True)

dstrat1_train_set = HuggingData(train_encodings, d_train_labels)
dstrat1_val_set = HuggingData(val_encodings, d_val_labels)

metric = evaluate.load("accuracy")

output = f"{dir}/models/results"
logs = f"{dir}/models/logs"

model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
training_args = TrainingArguments(
    output_dir=output,
    optim="adamw_torch",
    num_train_epochs=3,
    learning_rate= 1e-5,
    lr_scheduler_type = "cosine",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    save_strategy="no",
    warmup_steps= 10,
    weight_decay = 0.01,
    logging_dir=logs,
    logging_strategy="steps",
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=100
)

trainer = Trainer(
    model=model,                         
    args=training_args,                
    train_dataset=dstrat1_train_set,
    eval_dataset=dstrat1_val_set,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model(f"{dir}/models/")