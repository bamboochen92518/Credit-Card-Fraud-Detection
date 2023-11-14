from myfunc import ReadCSV, Sort_Data_with_CANO_and_Time, Tokenize_and_Normalize_Data, Data_to_Dataset, MyRNNModel, SaveParam
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from accelerate import Accelerator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--csv_file_path", type=str, default="../final/dataset_1st/training.csv")
parser.add_argument("--hidden_size", type=int, default=16)
parser.add_argument("--window_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--do_shuffle", type=bool, default=True)
parser.add_argument("--train_epoch", type=int, default=5)
parser.add_argument("--output_dir", type=str, default="1113")
args = parser.parse_args()

accelerator = Accelerator()

# Paremeters
numeric_features = ['locdt', 'loctm', 'conam', 'iterm', 'flam1', 'csmam']
category_features = ['chid', 'cano', 'contp', 'etymd', 'mchno', 'acqic', 'mcc', 'ecfg', 'insfg', 'bnsfg', 'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'flg_3dsmk']
num_numeric_features = len(numeric_features)
output_size = 1
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
vocab_file = f'{output_dir}/vocab.json'
param_file = f'{output_dir}/param.json'
SaveParam(args, param_file)

data = ReadCSV(args.csv_file_path)
tokenized_data, vocab_size = Tokenize_and_Normalize_Data(data, numeric_features, category_features, vocab_file, 'train')
sorted_data, txkey_list = Sort_Data_with_CANO_and_Time(tokenized_data, args.window_size) # list(list(dictionary * window size) * window)
vector_size = list()
for v in vocab_size:
    vector_size.append(min(16, v))

model = MyRNNModel(num_numeric_features, vocab_size, vector_size, args.hidden_size, output_size, args.window_size)
dataset = Data_to_Dataset(sorted_data, numeric_features, category_features, args.window_size)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.do_shuffle)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
#model = model.to(device)

'''
for x in dataloader:
    print(x)
    break
'''
# Loss function and optimizer
criterion = nn.BCELoss()  # Assuming binary classification
optimizer = Adam(model.parameters(), lr=args.learning_rate)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Training loop with tqdm
for epoch in range(args.train_epoch):
    tqdm_dataloader = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{args.train_epoch}', leave=True)
    for numeric_data, *category_data, labels in tqdm_dataloader:
        #numeric_data = numeric_data.to(device)
        #category_data = [category.to(device) for category in category_data]
        #labels = labels.to(device)
        optimizer.zero_grad()
        output = model(numeric_data, category_data)
        loss = criterion(output, labels.float())
        #loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        tqdm_dataloader.set_postfix({'Loss': loss.item()}, refresh=True)
    
    filename = f'{output_dir}/epoch{epoch + 1}'
    torch.save(model.state_dict(), filename)
