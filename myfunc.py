import csv
from tqdm import tqdm
from prettytable import PrettyTable

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
        for row in tqdm(data[1:], total=total_rows - 1, desc='Writing to CSV'):
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
        # return item['cano']

    # Sort the list of dictionaries based on 'age' and 'score'
    data = sorted(data, key=compare_func)
    cano_list = list()
    current_cano = ''
    current_list = list()
    for d in data:
        if d['cano'] != current_cano:
            if current_cano != '':
                cano_list.append([current_cano, current_list])
            current_cano = d['cano']
            current_list = [d]
        else:
            current_list.append(d)
    cano_list.append([current_cano, current_list])
    return cano_list
