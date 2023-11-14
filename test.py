from myfunc import ReadCSV, WriteCSV, Sort_Data_with_CANO_and_Time, Tokenize_and_Normalize_Data, Data_to_Dataset, MyRNNModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--csv_file_path", type=str, default="../final/dataset_1st/small_train.csv")
parser.add_argument("--model_path", type=str, default="1113_small")
parser.add_argument("--hidden_size", type=int, default=16)
parser.add_argument("--window_size", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--train_epoch", type=int, default=5)
parser.add_argument("--output_csv", type=str, default="prediction.csv")
args = parser.parse_args()

accelerator = Accelerator()

# Paremeters
numeric_features = ['locdt', 'loctm', 'conam', 'iterm', 'flam1', 'csmam']
category_features = ['chid', 'cano', 'contp', 'etymd', 'mchno', 'acqic', 'mcc', 'ecfg', 'insfg', 'bnsfg', 'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'flg_3dsmk']
num_numeric_features = len(numeric_features)
output_size = 1
vocab_file = f'{args.model_path}/vocab.json'

data = ReadCSV(args.csv_file_path)
tokenized_data, vocab_size = Tokenize_and_Normalize_Data(data, numeric_features, category_features, vocab_file, 'test')
sorted_data, txkey_list = Sort_Data_with_CANO_and_Time(tokenized_data, args.window_size) # list(list(dictionary * window size) * window)
vector_size = list()
for v in vocab_size:
    vector_size.append(min(16, v))

model = MyRNNModel(num_numeric_features, vocab_size, vector_size, args.hidden_size, output_size, args.window_size)
model.load_state_dict(torch.load(f'{args.model_path}/epoch{args.train_epoch}'))
dataset = Data_to_Dataset(sorted_data, numeric_features, category_features, args.window_size)
dataloader = DataLoader(dataset, batch_size=args.batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

'''
for x in dataloader:
    print(x)
    break
'''

model, dataloader = accelerator.prepare(model, dataloader)

# Training loop with tqdm
tqdm_dataloader = tqdm(dataloader, desc=f'Prediction', leave=True)
output_list = list()
for numeric_data, *category_data, labels in tqdm_dataloader:
    with torch.no_grad():
        output = model(numeric_data, category_data)
    output_list.append(output.item())

final_list = list()
for i in range(len(output_list)):
    final_list.append([txkey_list[i], output_list[i]])

WriteCSV(args.output_csv, final_list)
