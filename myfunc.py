import csv
from tqdm import tqdm
from prettytable import PrettyTable
import torch
from torch.utils.data import TensorDataset
from torch.nn.functional import normalize

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
        for row in tqdm(csv_reader, total=total_lines, desc='Reading from CSV'):
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
        for row in tqdm(data, total=total_rows, desc='Writing to CSV'):
            csv_writer.writerow(row)
    print("Finish!")
    return

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

def Sort_Data_with_CANO_and_Time(data):
    def compare_func(item):
        return item['cano'], int(item['locdt']), int(item['loctm'])

    data = sorted(data, key=compare_func)
    return data

def Tokenize_Data(data, numeric_features, catagory_features):
    key_dict = dict()
    for key in data[0].keys():
        key_dict[key] = {"count": 1}
    total_rows = len(data)
    for d in tqdm(data, total=total_rows, desc='Tokenize'):
        for k, v in d.items():
            if k == 'label':
                d[k] = int(d[k])
            if k in numeric_features:
                d[k] = float(d[k])
            if k in catagory_features:
                if v == '':
                    d[k] = 0
                else:
                    if v not in key_dict[k].keys():
                        key_dict[k][v] = key_dict[k]["count"]
                        key_dict[k]["count"] += 1
                    d[k] = key_dict[k][v]
    return data

def Calculate_Embedding_Dimension(data, category_features):
    max_dict = dict()
    for key in data[0].keys():
        if key in category_features:
            max_dict[key] = 0
    for d in data:
        for k, v in d.items():
            if k in category_features and v > max_dict[k]:
                max_dict[k] = v
    max_list = list()
    for v in max_dict.values():
        max_list.append(v + 1)
    return max_list

def Data_to_Dataset(tokenized_data, numeric_features, category_features):

    data = dict()
    n_input = list()
    c_input = list()
    for k in numeric_features + category_features + ['label']:
        data[k] = list()
    for d in tokenized_data:
        numeric_data = list()
        for k, v in d.items():
            if k in category_features + ['label']:
                data[k].append(v)
            if k in numeric_features:
                numeric_data.append(v)
        n_input.append(numeric_data)
    for k, v in data.items():
        if k in category_features:
            c_input.append(torch.tensor(v).view(-1, 1))
    n_input = torch.tensor(n_input)
    n_input = normalize(n_input, p=2.0)
    label = torch.tensor(data['label']).view(-1, 1)
    # Convert data to DataLoader
    return TensorDataset(n_input, *c_input, label)

