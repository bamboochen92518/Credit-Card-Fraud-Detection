'''
æè¢«çå·é->1
å¶ä»->0
'''
from myfunc import ReadCSV, ObserveKey, WriteCSV
train_file_path = '../final/dataset_1st/training.csv'
test_file_path = '../final/dataset_1st/public_processed.csv'
prediction = 'easy.csv'

train_data = ReadCSV(train_file_path)
val_dict = ObserveKey('cano', train_data, 1)
test_data = ReadCSV(test_file_path)
output = list()
for d in test_data:
    if d['cano'] in val_dict.keys() and val_dict[d['cano']][1] > 0:
        output.append([d['txkey'], 1])
    else:
        output.append([d['txkey'], 0])
WriteCSV(prediction, output)
