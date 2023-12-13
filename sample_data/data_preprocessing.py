import csv
from tqdm import tqdm
csv_train = '../dataset_1st/training.csv'
csv_valid = '../dataset_2nd/public.csv'
csv_test = '../dataset_2nd/private_1_processed.csv'
new_csv_train = 'final_train.csv'
new_csv_valid = 'final_valid.csv'
new_csv_test = 'final_test.csv'
numeric_features = ['locdt', 'loctm', 'conam', 'iterm', 'flam1', 'csmam']
category_features = ['chid', 'cano', 'contp', 'etymd', 'mchno', 'acqic', 'mcc', 'ecfg', 'insfg', 'bnsfg', 'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'flg_3dsmk']

train_data = list()
valid_data = list()
test_data = list()

train_lines = sum(1 for line in open(csv_train, 'r', encoding='utf-8'))
with open(csv_train, 'r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in tqdm(csv_reader, total=train_lines, desc='[Step 1] Reading Train Data', mininterval=1.0, miniters=100):
        train_data.append(row)

valid_lines = sum(1 for line in open(csv_valid, 'r', encoding='utf-8'))
with open(csv_valid, 'r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in tqdm(csv_reader, total=valid_lines, desc='[Step 2] Reading Validation Data', mininterval=1.0, miniters=100):
        valid_data.append(row)
test_lines = sum(1 for line in open(csv_test, 'r', encoding='utf-8'))
with open(csv_test, 'r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in tqdm(csv_reader, total=test_lines, desc='[Step 3] Reading Test Data', mininterval=1.0, miniters=100):
        test_data.append(row)
print(train_lines)
print(valid_lines)
print(test_lines)
merge_data = train_data + valid_data + test_data
def compare_func(item):
    return item['cano'], int(item['locdt']), int(item['loctm'])
print('[Step 4] Strat Sorting')
merge_data = sorted(merge_data, key=compare_func)
print("Finish Sorting")
length = len(merge_data)
for i in tqdm(range(length), total=length, desc='[Step 5] Adding day, week, month', mininterval=1.0, miniters=100):
    d = merge_data[i]
    current = i
    day = 0
    week = 0
    month = 0
    while merge_data[current]['cano'] == merge_data[i]['cano'] and current >= 0:
        if int(merge_data[i]['locdt']) - int(merge_data[current]['locdt']) < 1:
            day += 1
        if int(merge_data[i]['locdt']) - int(merge_data[current]['locdt']) < 7:
            week += 1
        if int(merge_data[i]['locdt']) - int(merge_data[current]['locdt']) < 30:
            month += 1
        current -= 1
    d['day'] = day
    d['week'] = week
    d['month'] = month
    merge_data[i] = d

value = dict()
for key in train_data[0].keys():
    value[key] = dict()
    if key in category_features:
        value[key][""] = [0, 0]

train_data = train_data + valid_data
length = len(train_data)
total = 0
for i in tqdm(range(length), total=length, desc='[Step 6] Observing each key', mininterval=1.0, miniters=100):
    d = train_data[i]
    for k, v in d.items():
        if k in category_features:
            if v not in value[k].keys():
                value[k][v] = [0, 0]
            if d['label'] == '0':
                value[k][v][0] += 1
            else:
                value[k][v][1] += 1
        if d['label'] == '1':
            total += 1

for d in tqdm(train_data, total=length, desc='[Step 7] Replacing each key in train data', mininterval=1.0, miniters=100):
    for k, v in list(d.items()):
        if k in category_features:
            p = 0
            if (value[k][v][0] + value[k][v][1]) != 0:
                p = value[k][v][1] / (value[k][v][0] + value[k][v][1])
            d[k] = p
    if 'label' in d.keys():
        tmp = d['label']
        del d['label']
        d['label'] = tmp

length = len(test_data) + len(valid_data)
for d in tqdm(test_data + valid_data, total=length, desc='[Step 8] Replacing each key in validation and test data', mininterval=1.0, miniters=100):
    for k, v in list(d.items()):
        if k in category_features:
            if v not in value[k].keys():
                v = ""
            p = 0
            if (value[k][v][0] + value[k][v][1]) != 0:
                p = value[k][v][1] / (value[k][v][0] + value[k][v][1])
            d[k] = p
    if 'label' in d.keys():
        tmp = d['label']
        del d['label']
        d['label'] = tmp
with open(new_csv_train, 'w', newline='', encoding='utf-8') as csv_file:
    fieldnames = train_data[0].keys()
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for d in train_data:
        writer.writerow(d)
with open(new_csv_valid, 'w', newline='', encoding='utf-8') as csv_file:
    fieldnames = valid_data[0].keys()
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for d in valid_data:
        writer.writerow(d)
test_data = valid_data + test_data
for d in test_data:
    if 'label' in d.keys():
        del d['label']
with open(new_csv_test, 'w', newline='', encoding='utf-8') as csv_file:
    fieldnames = test_data[0].keys()
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for d in test_data:
        writer.writerow(d)
