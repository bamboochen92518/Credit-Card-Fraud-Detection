import csv
from tqdm import tqdm
from prettytable import PrettyTable
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import copy
import json

def ReadCSV(csv_file_path):
    # Create a list to store dictionaries
    data = list()

    # Get the total number of lines in the CSV file
    total_lines = sum(1 for line in open(csv_file_path, 'r', encoding='utf-8'))

    # Open the CSV file
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
        # Create a CSV reader
        csv_reader = csv.DictReader(csv_file)

        # Iterate through rows in the CSV file with tqdm
        for row in tqdm(csv_reader, total=total_lines, desc='Reading from CSV', mininterval=1.0, miniters=100):
            data.append(row)
    return data

def WriteCSV(csv_file_path, data):
    total_rows = len(data)
    # Open the CSV file for writing
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        # Create a CSV writer
        csv_writer = csv.writer(csv_file)

        # Write the data to the CSV file
        header = ["txkey", "pred"]
        csv_writer.writerow(header)
        for row in tqdm(data, total=total_rows, desc='Writing to CSV', mininterval=1.0, miniters=100):
            csv_writer.writerow(row)
    print("Finish!")
    return

def SaveParam(args, param_file):
    param_dict = vars(args)
    with open(param_file, 'w') as f:
        json.dump(param_dict, f)

def ObserveKey(key, data, mode): # mode0: print table directly, mode1: write table into file
    value_set = set()
    value_dict = dict()
    for d in data:
        if d[key] not in value_set:
            value_set.add(d[key])
            if d["label"] == "0":
                value_dict[d[key]] = [1, 0]
            else:
                value_dict[d[key]] = [0, 1]
        else:
            if d["label"] == "0":
                value_dict[d[key]][0] += 1
            else:
                value_dict[d[key]][1] += 1

    # Create a PrettyTable object
    table = PrettyTable()

    # Define the table headers
    table.field_names = ["Key", "Total", "Normal", "Normal (%)", "Fake", "Fake (%)"]

    for k, v in value_dict.items():
        table.add_row([k, v[0] + v[1], v[0], v[0] / (v[0] + v[1]), v[1], v[1]/ (v[0] + v[1])])

    # Print the table
    if mode == 0:
        print(f'Observing {key}: ')
        print(table)
    if mode == 1:
        # Open the file for writing
        output_file_path = f'observe_result/{key}.txt'
        with open(output_file_path, 'w') as file:
            file.write(f'Observing {key}: \n')
            file.write(str(table))
    return value_dict

def Tokenize_and_Normalize_Data(data, numeric_features, catagory_features, vocab_file, mode):
    key_dict = dict()
    if mode == 'test':
        with open(vocab_file, 'r') as f:
            key_dict = json.load(f)
    if mode == 'train':
        for key in data[0].keys():
            if key in catagory_features:
                key_dict[key] = {"count": 1}
            if key in numeric_features:
                key_dict[key] = {"sum": 0, "square": 0}
    total_rows = len(data)
    for d in tqdm(data, total=total_rows, desc='Tokenize', mininterval=1.0, miniters=100):
        if mode == 'test':
            d['label'] = '0'
        for k, v in d.items():
            if k == 'label':
                d[k] = int(d[k])
            if k in numeric_features:
                d[k] = float(d[k])
                if mode == 'train':
                    key_dict[k]["sum"] += d[k]
                    key_dict[k]["square"] += pow(d[k], 2)
            if k in catagory_features:
                if v == '':
                    d[k] = 0
                else:
                    if v not in key_dict[k].keys():
                        if mode == 'train':
                            key_dict[k][v] = key_dict[k]["count"]
                            key_dict[k]["count"] += 1
                            d[k] = key_dict[k][v]
                        if mode == 'test':
                            d[k] = 0
                    else:
                        d[k] = key_dict[k][v]
    if mode == 'train': 
        for key in data[0].keys():
            if key in numeric_features:
                key_dict[key]["mean"] = key_dict[key]["sum"] / total_rows
                key_dict[key]["stderr"] = pow(((key_dict[key]["square"] - pow(key_dict[key]["mean"], 2)) / total_rows), 0.5)

    for d in tqdm(data, total=total_rows, desc='Normalize', mininterval=1.0, miniters=100):
        for k, v in d.items():
            if k in numeric_features:
                if key_dict[k]["stderr"] != 0:
                    d[k] = (d[k] - key_dict[k]["mean"]) / key_dict[k]["stderr"]
                else:
                    d[k] = 0
    
    embedding_dims = list()
    for k, v in list(key_dict.items()):
        if k in catagory_features:
            embedding_dims.append(v["count"])
    
    if mode == 'train':
        with open(vocab_file, 'w') as file:
            json.dump(key_dict, file)

    return data, embedding_dims

def Sort_Data_with_CANO_and_Time(data, window_size):
    def compare_func(item):
        return item['cano'], int(item['locdt']), int(item['loctm'])

    data = sorted(data, key=compare_func)
    empty_data = copy.deepcopy(data[0])
    for k in empty_data.keys():
        empty_data[k] = 0

    CANO_data = list()
    ID = list()
    current_cano = ''
    current_list = list()
    counter = 0
    total_rows = len(data)
    for d in tqdm(data, total=total_rows, desc='Grouping', mininterval=1.0, miniters=100):
        if d['cano'] == current_cano and counter < window_size:
            current_list.append(d)
            counter += 1
        else:
            if d['cano'] != current_cano:
                current_list = [d]
                current_cano = d['cano']
                counter = 1
            else:
                current_list = current_list[1:] + [d]

        CANO_data.append([empty_data] * (window_size - counter) + current_list)
        ID.append(d['txkey'])

    return CANO_data, ID

def Data_to_Dataset(sorted_data, numeric_features, category_features, window_size):

    data = dict()
    n_input = list()
    c_input = list()
    for k in category_features + ['label']:
        data[k] = list()
        for i in range(window_size):
            data[k].append(list())

    total_rows = len(sorted_data)
    for s in tqdm(sorted_data, total=total_rows, desc='Loading Dataset', mininterval=1.0, miniters=100):
        numeric_data = list()
        for i in range(len(s)):
            d = s[i]
            for k, v in d.items():
                if k in category_features + ['label']:
                    data[k][i].append(v)
                if k in numeric_features:
                    numeric_data.append(v)
        n_input.append(numeric_data)
    for i in range(window_size):
        for k, v in data.items():
            if k in category_features:
                c_input.append(torch.tensor(v[i]).view(-1, 1))
    n_input = torch.tensor(n_input)
    label = torch.tensor(data['label'][-1]).view(-1, 1)
    # Convert data to DataLoader
    return TensorDataset(n_input, *c_input, label)

class MyRNNModel(nn.Module):
    def __init__(self, num_numeric_features, vocab_size, vector_size, hidden_size, output_size, window_size):
        super(MyRNNModel, self).__init__()
        self.num_numeric_features = num_numeric_features
        self.window_size = window_size
        # Embedding 层
        self.embeddings = nn.ModuleList()
        for num_categories, embedding_dim in zip(vocab_size, vector_size):
            try:
                embedding_layer = nn.Embedding(num_categories, embedding_dim)
                self.embeddings.append(embedding_layer)
            except IndexError as e:
                print(f"Error: {e}, num_categories: {num_categories}")
                raise e
        # LSTM Layer
        self.rnn = nn.LSTM(input_size=num_numeric_features + sum(vector_size), hidden_size=hidden_size, batch_first=True)
        # ReLU Layer
        self.relu = nn.ReLU()
        # Fully Connected Layer
        self.fc_output = nn.Linear(hidden_size, output_size)
        # Sigmoid Layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, numeric_data, category_data):
        numeric_data = numeric_data.tolist()
        batch_size = len(numeric_data)
        final_numeric_data = list()
        for w in range(self.window_size):
            split_numeric_data = list()
            for b in range(batch_size):
                split_numeric_data.append(numeric_data[b][w*(self.num_numeric_features):(w+1)*(self.num_numeric_features)])
            split_numeric_data = torch.tensor(split_numeric_data).cuda()
            final_numeric_data.append(split_numeric_data)
        num_categories = int(len(category_data) / self.window_size)
        category_data = [category_data[i:i+num_categories] for i in range(0, len(category_data), num_categories)]
       
        combined_features = final_numeric_data[0].unsqueeze(1)
        numeric_data = list()
        for i in range(self.window_size):
            # Embedding
            embedded_categories = []
            for embedding, category in zip(self.embeddings, category_data[i]):
                try:
                    embedded_category = embedding(category)
                    embedded_categories.append(embedded_category)
                except IndexError as e:
                    print(f"Error: {e}, embedding_dim: {embedding.embedding_dim}, num_categories: {embedding.num_embeddings}")
                    print("category:", category)
                    raise e
            # print("Embedding Category: ")
            # print(embedded_categories)
            # [tensor([[[vector size]]], grad_fn=...)] * (category num + 1)
            embedded_categories_combined = torch.cat(embedded_categories, dim=-1)
            # print("Embedding Category Combined: ")
            # print(embedded_categories_combined)
            # [tensor([[[sum(vector size)]]])]
            
            # Concatenate numeric and embedded category features
            combined_features = torch.cat([final_numeric_data[i].unsqueeze(1), embedded_categories_combined], dim=-1)
            numeric_data.append(combined_features)
        
        combined_features = torch.cat(numeric_data, dim=1)
        # print(f"Shape of combined_features: {combined_features.shape}")
        # print("Combined Feature: ")
        # print(combined_features)
        # [tensor[[[sum(vector size) + numeric num]]]]

        # RNN forward pass
        rnn_output, _ = self.rnn(combined_features)
        # print(f"Shape of rnn_output: {rnn_output.shape}")
        relu_output = self.relu(rnn_output[:, -1, :])
        # print(f"Shape of relu_output: {relu_output.shape}")
        fc_output = self.fc_output(relu_output)
        # print(f"Shape of fc_output: {fc_output.shape}")
        output = self.sigmoid(fc_output)
        # print(f"Shape of output: {output.shape}")

        # Fully connected layer for prediction
        return output

