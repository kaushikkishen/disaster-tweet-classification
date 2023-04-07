import sys
import numpy as np 
import pandas as pd
import torch
import transformers
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from tqdm import tqdm
device = 'cuda:0' if cuda.is_available() else 'cpu'
from transformers import pipeline

def data_iterator(df):
    for i in range(df.shape[0]):
        yield str(df.text[i])

def get_predict(iterator, pipeline):
    df = pd.DataFrame(columns = ['label', 'score'])
    for idx, out in enumerate(tqdm(pipeline(iterator))):
        df.loc[idx] = out
    return df

def clean_submit(preds):
    df = preds.copy()
    df.reset_index(inplace=True)
    df.rename(columns = {'index':'id'}, inplace = True)
    df.set_index("id", inplace=True)
    df['target'] = df.apply(lambda x: 1 if x['label'] == 'POSITIVE' else 0, axis = 1)
    df.drop(columns=['score', 'label'], inplace=True)
    return df

dir = sys.argv[1]
d_train = pd.read_csv(f"{dir}/train.csv")
d_test = pd.read_csv(f"{dir}/test.csv")

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
disaster = pipeline("text-classification", model=f"{dir}/trainer_results", 
                    tokenizer = tokenizer, device=device, function_to_apply="softmax")


preds_train = get_predict(data_iterator(d_train), disaster)
preds_test = get_predict(data_iterator(d_test), disaster)

submit_train = clean_submit(preds_train)
submit_test = clean_submit(preds_test)

d_test.reset_index(inplace=True)
submit_test.reset_index(inplace=True)
test = d_test[["id"]].copy()
merged = test.merge(submit_test, left_index=True, right_index=True, suffixes=(None, "_s"))
merged.drop(labels=["id_s"],axis=1,inplace=True)
merged.set_index("id",inplace=True)

submit_train.to_csv("submit_train.csv")
merged.to_csv("submit_test.csv")